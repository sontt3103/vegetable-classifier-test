"""Inference cho model SVM cổ điển: ảnh -> HOG + Color Histogram -> (PCA) -> SVM.

Cùng interface với `VegetableClassifier` trong predict.py (predict / model_info,
trả về PredictionResult) nên Streamlit dùng được cả 3 model.

Hỗ trợ 2 định dạng file model:
  1. BUNDLE do `python -m src.training.train_svm` lưu: checkpoints/classical_svm.pkl
     (dict gồm backend, pipeline hoặc pca+svm, class_names, feature_params...).
     Đây là định dạng mặc định hiện tại của nhóm.
  2. CẶP FILE do classical_svm.py / train_svm --backend tn1 lưu:
     checkpoints/svm_model.pkl + checkpoints/pca_model.pkl.

LƯU Ý — tránh train/serving skew:
    Tiền xử lý của SVM KHÁC CNN (CNN: resize 224 + chia 255). SVM: resize 128
    (cv2.INTER_AREA), giữ 0-255, rồi `extract_features_from_image` của
    hog_features.py (HOG + histogram màu x COLOR_WEIGHT) — đúng hàm lúc train gọi.
    Vì vậy KHÔNG dùng `load_image_for_inference` của predict.py cho SVM.
    Với bundle, các tham số đặc trưng lúc train được lưu kèm; nếu hog_features.py
    hiện tại khác (vd đổi COLOR_WEIGHT) thì báo lỗi rõ ràng thay vì dự đoán sai âm thầm.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from configs.config import CONFIG
from src.inference.predict import InvalidImageError, PredictionResult

BUNDLE_NAME = "classical_svm.pkl"      # train_svm.py
SVM_CHECKPOINT_NAME = "svm_model.pkl"  # classical_svm.py / train_svm --backend tn1
PCA_CHECKPOINT_NAME = "pca_model.pkl"


def find_svm_checkpoint(ckpt_dir) -> Path | None:
    """Trả về file model SVM dùng được trong ckpt_dir, hoặc None nếu chưa có.
    Ưu tiên bundle (classical_svm.pkl); nếu không có thì cần đủ svm_model.pkl + pca_model.pkl."""
    ckpt_dir = Path(ckpt_dir)
    bundle = ckpt_dir / BUNDLE_NAME
    if bundle.exists():
        return bundle
    svm, pca = ckpt_dir / SVM_CHECKPOINT_NAME, ckpt_dir / PCA_CHECKPOINT_NAME
    if svm.exists() and pca.exists():
        return svm
    return None


def _to_rgb_uint8(image_source, img_size: tuple) -> np.ndarray:
    """path / file-like / PIL.Image -> ndarray RGB uint8 shape (H, W, 3)."""
    try:
        img = image_source if isinstance(image_source, Image.Image) else Image.open(image_source)
        img = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise InvalidImageError("File không phải là ảnh hợp lệ (không đọc được định dạng ảnh).") from e
    except Exception as e:  # noqa: BLE001
        raise InvalidImageError(f"Không thể đọc ảnh: {e}") from e

    array = np.asarray(img, dtype=np.uint8)
    # Giống load_and_resize của hog_features.py: size (width, height), INTER_AREA
    return cv2.resize(array, tuple(img_size), interpolation=cv2.INTER_AREA)


def _ensure_pickle_compat():
    """Cho phép load file .pkl được lưu khi chạy `python -m src.models.classical_svm`.

    Khi chạy bằng `-m`, class ClassicalSVMClassifier bị pickle với tên module là
    `__main__`, nên process khác (Streamlit) không tìm thấy class để load. Gắn class
    vào `__main__` để joblib.load tìm được.
    """
    import __main__

    from src.models.classical_svm import ClassicalSVMClassifier

    if not hasattr(__main__, "ClassicalSVMClassifier"):
        __main__.ClassicalSVMClassifier = ClassicalSVMClassifier


def _load_pca(path: Path):
    """Load PCA. Ưu tiên PCAWrapper.load (TN 2); nếu không có thì joblib.load."""
    try:
        from src.features.pca_wrapper import PCAWrapper

        if hasattr(PCAWrapper, "load"):
            return PCAWrapper.load(str(path))
    except ImportError:
        pass
    return joblib.load(path)


def _current_feature_params() -> dict:
    """Tham số đặc trưng hiện tại của hog_features.py (cùng cách train_svm.py ghi lại)."""
    from src.features import hog_features as hf

    params = {
        "img_size": list(hf.DEFAULT_IMG_SIZE),
        "hog_orientations": hf.HOG_ORIENTATIONS,
        "hog_pixels_per_cell": list(hf.HOG_PIXELS_PER_CELL),
        "hog_cells_per_block": list(hf.HOG_CELLS_PER_BLOCK),
        "color_hist_bins": hf.COLOR_HIST_BINS,
        "color_weight": getattr(hf, "COLOR_WEIGHT", 1.0),
    }
    return json.loads(json.dumps(params))  # tuple -> list để so sánh được


def _check_feature_params(saved: dict) -> None:
    current = _current_feature_params()
    saved = json.loads(json.dumps(saved))
    diffs = {k: (saved[k], current[k]) for k in saved if k in current and saved[k] != current[k]}
    if diffs:
        raise ValueError(
            "Model SVM được train với tham số đặc trưng khác hog_features.py hiện tại "
            f"(lúc train → hiện tại): {diffs}. Hãy dùng đúng bản hog_features.py lúc train "
            "hoặc train lại SVM, nếu không dự đoán sẽ sai mà không báo lỗi."
        )


class SVMVegetableClassifier:
    def __init__(
        self,
        checkpoint_path: str,
        class_names: list[str] | None = None,
        pca_path: str | None = None,
        config: dict = CONFIG,
    ):
        self.config = config
        self.checkpoint_path = str(checkpoint_path)
        ckpt_dir = Path(config["paths"]["checkpoints_dir"])

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Không thấy file SVM: {self.checkpoint_path}")

        from src.features.hog_features import extract_features_from_image

        self._extract = extract_features_from_image

        _ensure_pickle_compat()
        obj = joblib.load(self.checkpoint_path)

        self.bundle = None
        self.svm = None
        self.pca = None
        self.pca_path = None
        if isinstance(obj, dict) and "backend" in obj:
            # --- Định dạng 1: bundle của train_svm.py ---
            self.bundle = obj
            if obj.get("feature_params"):
                _check_feature_params(obj["feature_params"])
            fp = obj.get("feature_params") or {}
            default_size = config.get("svm", {}).get("img_size", (128, 128))
            self.img_size = tuple(fp.get("img_size", default_size))
            if obj["backend"] == "sklearn":
                self._estimator = obj["pipeline"]
            else:
                self._estimator = obj["svm"].model
            class_names = class_names or obj.get("class_names")
        else:
            # --- Định dạng 2: svm_model.pkl + pca_model.pkl ---
            self.pca_path = Path(pca_path) if pca_path else ckpt_dir / PCA_CHECKPOINT_NAME
            if not self.pca_path.exists():
                raise FileNotFoundError(
                    f"Không thấy file PCA: {self.pca_path}. SVM cần đúng bộ PCA đã fit lúc train."
                )
            self.svm = obj
            self.pca = _load_pca(self.pca_path)
            self._estimator = getattr(obj, "model", obj)
            from src.features.hog_features import DEFAULT_IMG_SIZE

            self.img_size = tuple(config.get("svm", {}).get("img_size", DEFAULT_IMG_SIZE))

        self.class_names = class_names or self._resolve_class_names()
        if not self.class_names:
            raise ValueError("Không xác định được danh sách class cho SVM.")

        n_cls = len(getattr(self._estimator, "classes_", []))
        if n_cls and n_cls != len(self.class_names):
            raise ValueError(
                f"SVM có {n_cls} class nhưng danh sách tên class có {len(self.class_names)}."
            )

    # ------------------------------------------------------------------
    def _resolve_class_names(self) -> list[str] | None:
        """Thứ tự ưu tiên: svm.classes -> metadata -> tên thư mục train."""
        classes = getattr(self.svm, "classes", None)
        if classes is not None and len(classes) > 0:
            return [str(c) for c in classes]

        reports = Path(self.config["paths"]["reports_dir"])
        for stem in ("classical_svm", "mobilenetv2_transfer", "baseline_cnn"):
            meta = reports / f"{stem}_metadata.json"
            if meta.exists():
                with open(meta, encoding="utf-8") as f:
                    names = json.load(f).get("class_names")
                if names:
                    return names

        train_dir = Path(self.config["paths"]["train_dir"])
        if train_dir.exists():
            return sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        return None

    def _features(self, image_source) -> np.ndarray:
        image = _to_rgb_uint8(image_source, self.img_size)
        return self._extract(image).reshape(1, -1)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.bundle is not None:
            # Dùng đúng hàm của train_svm.py để không lệch với lúc đánh giá
            from src.training.train_svm import bundle_predict_proba

            return bundle_predict_proba(self.bundle, X)[0]
        return self._estimator.predict_proba(self.pca.transform(X))[0]

    # ------------------------------------------------------------------
    def predict(self, image_source, top_k: int | None = None) -> PredictionResult:
        top_k = top_k or self.config["inference"]["top_k"]

        probs = self._predict_proba(self._features(image_source))

        # Cột của predict_proba tương ứng với estimator.classes_ (nhãn số đã sort)
        labels = getattr(self._estimator, "classes_", np.arange(len(probs)))

        top_k = min(top_k, len(probs))
        order = np.argsort(probs)[::-1][:top_k]
        results = [
            {"class": self.class_names[int(labels[i])], "probability": float(probs[i])}
            for i in order
        ]
        return PredictionResult(
            predicted_class=results[0]["class"],
            confidence=results[0]["probability"],
            top_k=results,
        )

    # ------------------------------------------------------------------
    def _inner_svc(self):
        """SVC thật: trong Pipeline (bundle sklearn) hoặc CalibratedClassifierCV (code TN 1)."""
        est = self._estimator
        try:
            return est.named_steps["svc"]
        except (AttributeError, KeyError):
            pass
        try:
            return est.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            return est

    def _read_metadata(self) -> dict:
        meta = Path(self.config["paths"]["reports_dir"]) / "classical_svm_metadata.json"
        if meta.exists():
            try:
                with open(meta, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def model_info(self) -> dict:
        svc = self._inner_svc()
        support = getattr(svc, "n_support_", None)
        n_support = int(np.sum(support)) if support is not None else None
        meta = self._read_metadata()

        pca_components, best_params = None, None
        if self.bundle is not None:
            n_support = self.bundle.get("n_support_vectors") or n_support
            if self.bundle["backend"] == "sklearn":
                try:
                    pca_components = self.bundle["pipeline"].named_steps["features"] \
                        .named_transformers_["hog"].n_components_
                except Exception:  # noqa: BLE001
                    pass
                best_params = meta.get("best_params")
            else:
                pca_components = getattr(self.bundle["pca"], "n_components_", None)
                best_params = getattr(self.bundle["svm"], "best_params", None)
        else:
            pca_components = getattr(getattr(self.pca, "pca", self.pca), "n_components_", None)
            best_params = getattr(self.svm, "best_params", None)

        return {
            "kind": "svm",
            "checkpoint_path": self.checkpoint_path,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_size": self.img_size[0],
            "num_parameters": None,  # SVM không có "số parameters" như mạng nơ-ron
            "num_support_vectors": n_support,
            "pca_components": pca_components,
            "best_params": best_params,
            "kernel": getattr(svc, "kernel", None),
            "architecture": meta.get("architecture"),
        }
