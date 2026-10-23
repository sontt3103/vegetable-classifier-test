"""Inference cho model SVM cổ điển: ảnh -> HOG + Color Histogram -> (PCA) -> SVM.

Cùng interface với `VegetableClassifier` trong predict.py (predict / model_info,
trả về PredictionResult) nên Streamlit dùng được cả 3 model mà không phải rẽ
nhánh phức tạp.

Đọc được 3 kiểu file checkpoint:
    1. checkpoints/classical_svm.pkl  (do src/training/train_svm.py sinh ra — MẶC ĐỊNH)
         bundle dict {"backend": "sklearn", "pipeline": Pipeline(...), "class_names": ...}
         backend "sklearn": pipeline đã gồm PCA + màu Hellinger + SVC -> đưa thẳng vector
                            8196 chiều vào pipeline, KHÔNG áp PCA riêng.
         backend "tn1"    : bundle["pca"] (PCAWrapper) + bundle["svm"] (ClassicalSVMClassifier)
    2. checkpoints/svm_model.pkl + pca_model.pkl  (src/models/classical_svm.py của TN 1)

LƯU Ý QUAN TRỌNG — tránh train/serving skew:
    Tiền xử lý của SVM KHÁC CNN. CNN: resize 224 + chia 255. SVM: resize 128
    (cv2.INTER_AREA), giữ giá trị 0-255 (uint8), rồi mới tính HOG + color histogram
    (phần màu nhân COLOR_WEIGHT). Vì vậy KHÔNG dùng `load_image_for_inference`
    của predict.py cho SVM. Ở đây gọi `extract_features_from_image` của
    hog_features.py — đúng hàm mà lúc train (extract_combined_features) cũng gọi.
    Với bundle của train_svm.py, tham số HOG lúc train được lưu trong bundle và
    được so sánh với hog_features.py hiện tại; lệch thì báo lỗi rõ ràng thay vì
    dự đoán sai âm thầm.
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

BUNDLE_NAME = "classical_svm.pkl"
LEGACY_SVM_NAME = "svm_model.pkl"
LEGACY_PCA_NAME = "pca_model.pkl"


def find_svm_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Trả về file SVM dùng được trong thư mục checkpoints, hoặc None nếu chưa có."""
    ckpt_dir = Path(ckpt_dir)
    bundle = ckpt_dir / BUNDLE_NAME
    if bundle.exists():
        return bundle
    legacy = ckpt_dir / LEGACY_SVM_NAME
    if legacy.exists() and (ckpt_dir / LEGACY_PCA_NAME).exists():
        return legacy
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
    vào `__main__` để joblib.load tìm được. (Bundle của train_svm.py đã tự tránh lỗi này.)
    """
    import __main__

    from src.models.classical_svm import ClassicalSVMClassifier

    if not hasattr(__main__, "ClassicalSVMClassifier"):
        __main__.ClassicalSVMClassifier = ClassicalSVMClassifier


def _current_feature_params() -> dict:
    """Tham số trích đặc trưng hiện tại — cùng cấu trúc với _feature_params() của train_svm.py."""
    from src.features import hog_features as hf

    return {
        "img_size": list(hf.DEFAULT_IMG_SIZE),
        "hog_orientations": hf.HOG_ORIENTATIONS,
        "hog_pixels_per_cell": list(hf.HOG_PIXELS_PER_CELL),
        "hog_cells_per_block": list(hf.HOG_CELLS_PER_BLOCK),
        "color_hist_bins": hf.COLOR_HIST_BINS,
        "color_weight": getattr(hf, "COLOR_WEIGHT", 1.0),
    }


def _load_pca(path: Path):
    """Load PCA. Ưu tiên PCAWrapper.load (TN 2); nếu không có thì joblib.load."""
    try:
        from src.features.pca_wrapper import PCAWrapper

        if hasattr(PCAWrapper, "load"):
            return PCAWrapper.load(str(path))
    except ImportError:
        pass
    return joblib.load(path)


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
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Không thấy file SVM: {self.checkpoint_path}")

        # Import ở đây để lỗi (nếu TN 2 chưa xong) báo rõ ràng, không làm hỏng import cả app
        from src.features.hog_features import DEFAULT_IMG_SIZE, extract_features_from_image

        self._extract = extract_features_from_image
        self.img_size = tuple(config.get("svm", {}).get("img_size", DEFAULT_IMG_SIZE))

        _ensure_pickle_compat()
        obj = joblib.load(self.checkpoint_path)

        self.bundle = obj if isinstance(obj, dict) and "backend" in obj else None
        self.pca = None
        if self.bundle is not None:
            self._init_from_bundle(self.bundle)
        else:
            self._init_legacy(obj, pca_path)

        self.class_names = class_names or self._resolve_class_names()
        if not self.class_names:
            raise ValueError("Không xác định được danh sách class cho SVM.")
        if len(self._labels) and len(self._labels) != len(self.class_names):
            raise ValueError(
                f"SVM có {len(self._labels)} class nhưng danh sách tên class có {len(self.class_names)}."
            )

    # ------------------------------------------------------------------
    def _init_from_bundle(self, b: dict) -> None:
        saved = b.get("feature_params")
        if saved:
            now = _current_feature_params()
            diff = {k: (saved.get(k), now.get(k)) for k in now
                    if json.dumps(saved.get(k)) != json.dumps(now.get(k))}
            if diff:
                raise ValueError(
                    "Tham số trích đặc trưng trong hog_features.py KHÁC lúc train SVM "
                    f"(lúc train, hiện tại): {diff}. Train lại SVM hoặc đưa tham số về như cũ, "
                    "nếu không dự đoán sẽ sai."
                )
            self.img_size = tuple(saved["img_size"])

        if b["backend"] == "sklearn":
            pipe = b["pipeline"]
            self._proba = pipe.predict_proba
            self._labels = np.asarray(pipe.classes_)
            self._svc = pipe.named_steps.get("svc", pipe)
        elif b["backend"] == "tn1":
            self.pca, svm = b["pca"], b["svm"]
            est = getattr(svm, "model", svm)
            self._proba = lambda X: est.predict_proba(self.pca.transform(X))
            self._labels = np.asarray(est.classes_)
            self._svc = self._inner_svc(est)
        else:
            raise ValueError(f"Backend SVM không hỗ trợ: {b['backend']}")
        self._class_names_saved = b.get("class_names")

    def _init_legacy(self, svm, pca_path) -> None:
        pca_file = Path(pca_path) if pca_path else Path(self.checkpoint_path).with_name(LEGACY_PCA_NAME)
        if not pca_file.exists():
            raise FileNotFoundError(
                f"Không thấy file PCA: {pca_file}. SVM cần đúng bộ PCA đã fit lúc train."
            )
        self.pca = _load_pca(pca_file)
        self._legacy_svm = svm
        est = getattr(svm, "model", svm)
        self._proba = lambda X: est.predict_proba(self.pca.transform(X))
        self._labels = np.asarray(getattr(est, "classes_", []))
        self._svc = self._inner_svc(est)
        self._class_names_saved = getattr(svm, "classes", None)

    @staticmethod
    def _inner_svc(est):
        """SVC thật nằm bên trong CalibratedClassifierCV (nếu có) — để đọc n_support_, kernel."""
        try:
            return est.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            return est

    # ------------------------------------------------------------------
    def _resolve_class_names(self) -> list[str] | None:
        """Thứ tự ưu tiên: tên class lưu trong checkpoint -> metadata CNN -> tên thư mục train."""
        saved = self._class_names_saved
        if saved is not None and len(saved) > 0:
            return [str(c) for c in saved]

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

    # ------------------------------------------------------------------
    def predict(self, image_source, top_k: int | None = None) -> PredictionResult:
        top_k = top_k or self.config["inference"]["top_k"]

        probs = self._proba(self._features(image_source))[0]
        labels = self._labels if len(self._labels) else np.arange(len(probs))

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

    def _metadata(self) -> dict:
        """reports/classical_svm_metadata.json (nếu có) — lấy best_params, kiến trúc, số chiều PCA."""
        reports = Path(self.config["paths"]["reports_dir"])
        for stem in (Path(self.checkpoint_path).stem, "classical_svm"):
            p = reports / f"{stem}_metadata.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def model_info(self) -> dict:
        meta = self._metadata()
        svc = self._svc
        support = getattr(svc, "n_support_", None)
        n_sv = None
        if self.bundle is not None and self.bundle.get("n_support_vectors"):
            n_sv = int(self.bundle["n_support_vectors"])
        elif support is not None:
            n_sv = int(np.sum(support))

        pca_components = meta.get("pca_components")
        if pca_components is None and self.pca is not None:
            pca_components = getattr(getattr(self.pca, "pca", self.pca), "n_components_", None)

        return {
            "kind": "svm",
            "checkpoint_path": self.checkpoint_path,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_size": self.img_size[0],
            "num_parameters": None,  # SVM không có "số parameters" như mạng nơ-ron
            "num_support_vectors": n_sv,
            "pca_components": pca_components,
            "best_params": meta.get("best_params") or getattr(getattr(self, "_legacy_svm", None), "best_params", None),
            "kernel": getattr(svc, "kernel", None),
            "architecture": meta.get("architecture"),
        }
