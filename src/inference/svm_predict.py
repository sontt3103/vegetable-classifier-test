"""Inference cho model SVM cổ điển: ảnh -> HOG + Color Histogram -> PCA -> SVM.

Cùng interface với `VegetableClassifier` trong predict.py (predict / model_info,
trả về PredictionResult) nên Streamlit dùng được cả 3 model mà không phải rẽ
nhánh phức tạp.

LƯU Ý QUAN TRỌNG — tránh train/serving skew:
    Tiền xử lý của SVM KHÁC CNN. CNN: resize 224 + chia 255. SVM: resize 128
    (cv2.INTER_AREA), giữ giá trị 0-255 (uint8), rồi mới tính HOG + color histogram
    (phần màu nhân COLOR_WEIGHT). Vì vậy KHÔNG dùng `load_image_for_inference`
    của predict.py cho SVM. Ở đây gọi `extract_features_from_image` của
    hog_features.py — đúng hàm mà lúc train (extract_combined_features) cũng gọi —
    để đặc trưng lúc dự đoán giống hệt lúc train.
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

SVM_CHECKPOINT_NAME = "svm_model.pkl"
PCA_CHECKPOINT_NAME = "pca_model.pkl"


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
    vào `__main__` để joblib.load tìm được. Cách sửa gốc: xem ghi chú trong
    classical_svm.py (import lại module ở khối `if __name__ == "__main__"`).
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
        self.pca_path = Path(pca_path) if pca_path else ckpt_dir / PCA_CHECKPOINT_NAME

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Không thấy file SVM: {self.checkpoint_path}")
        if not self.pca_path.exists():
            raise FileNotFoundError(
                f"Không thấy file PCA: {self.pca_path}. SVM cần đúng bộ PCA đã fit lúc train."
            )

        # Import ở đây để lỗi (nếu TN 2 chưa xong) báo rõ ràng, không làm hỏng import cả app
        from src.features.hog_features import DEFAULT_IMG_SIZE, extract_features_from_image

        self._extract = extract_features_from_image
        # Kích thước ảnh phải khớp lúc train (classical_svm.py lấy từ CONFIG["svm"]["img_size"])
        self.img_size = tuple(config.get("svm", {}).get("img_size", DEFAULT_IMG_SIZE))

        _ensure_pickle_compat()
        self.svm = joblib.load(self.checkpoint_path)  # ClassicalSVMClassifier đã train
        self.pca = _load_pca(self.pca_path)
        # SVC bên trong; fallback nếu file .pkl lưu thẳng SVC / Pipeline
        self._estimator = getattr(self.svm, "model", self.svm)

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
        """Thứ tự ưu tiên: svm.classes -> metadata của CNN/MobileNetV2 -> tên thư mục train."""
        classes = getattr(self.svm, "classes", None)
        if classes is not None and len(classes) > 0:
            return [str(c) for c in classes]

        reports = Path(self.config["paths"]["reports_dir"])
        for stem in ("mobilenetv2_transfer", "baseline_cnn"):
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

        feat = self.pca.transform(self._features(image_source))
        probs = self._estimator.predict_proba(feat)[0]

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

    def _inner_svc(self):
        """SVC thật nằm bên trong CalibratedClassifierCV (nếu có) — để đọc n_support_, kernel."""
        est = self._estimator
        try:
            return est.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            return est

    def model_info(self) -> dict:
        est = self._inner_svc()
        support = getattr(est, "n_support_", None)
        return {
            "kind": "svm",
            "checkpoint_path": self.checkpoint_path,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_size": self.img_size[0],
            "num_parameters": None,  # SVM không có "số parameters" như mạng nơ-ron
            "num_support_vectors": int(np.sum(support)) if support is not None else None,
            "pca_components": getattr(getattr(self.pca, "pca", self.pca), "n_components_", None),
            "best_params": getattr(self.svm, "best_params", None),
            "kernel": getattr(est, "kernel", None),
        }
