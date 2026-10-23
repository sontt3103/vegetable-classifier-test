"""Inference pipeline: 1 ảnh bất kỳ -> preprocessing -> model -> prediction + confidence.

Đây là pipeline DUY NHẤT dùng chung cho cả test script và Streamlit demo app,
để đảm bảo app luôn dùng đúng cách preprocessing giống lúc train/eval (tránh
train/serving skew — 1 lỗi rất phổ biến khi làm demo ML).

Luồng xử lý:
    Input Image -> Preprocessing (resize + normalize) -> Trained Model
    -> Prediction -> Vegetable/Fruit Class -> Confidence Score
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from configs.config import CONFIG


class InvalidImageError(Exception):
    """Raised khi ảnh input không hợp lệ (không mở được, sai định dạng, v.v.)."""


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    top_k: list[dict]  # [{"class": str, "probability": float}, ...]

    def to_dict(self):
        return asdict(self)


def load_image_for_inference(image_source, img_size: int) -> np.ndarray:
    """Load ảnh từ path/bytes/PIL.Image, resize, normalize về [0, 1].

    Ném InvalidImageError với message rõ ràng nếu ảnh hỏng/sai định dạng,
    để app phía trên (Streamlit) có thể hiển thị lỗi thân thiện cho người dùng
    thay vì crash.
    """
    try:
        if isinstance(image_source, Image.Image):
            img = image_source
        else:
            img = Image.open(image_source)
        img = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise InvalidImageError("File không phải là ảnh hợp lệ (không đọc được định dạng ảnh).") from e
    except Exception as e:  # noqa: BLE001
        raise InvalidImageError(f"Không thể đọc ảnh: {e}") from e

    img = img.resize((img_size, img_size))
    array = np.asarray(img, dtype=np.float32) / 255.0
    return array


class VegetableClassifier:
    """Wrapper load model 1 lần, dùng lại cho nhiều lần predict (tránh load lại
    model mỗi request — quan trọng cho performance của Streamlit app).
    """

    def __init__(self, checkpoint_path: str, class_names: list[str] | None = None, config: dict = CONFIG):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.model = tf.keras.models.load_model(checkpoint_path)
        self.class_names = class_names or self._load_class_names_from_metadata()
        if not self.class_names:
            raise ValueError(
                "Không xác định được danh sách class. Hãy truyền class_names hoặc "
                "đảm bảo file metadata JSON tương ứng tồn tại trong reports/."
            )

    def _load_class_names_from_metadata(self) -> list[str] | None:
        """Cố gắng tự động lấy class_names từ file metadata JSON được lưu lúc train,
        theo tên checkpoint (vd checkpoints/baseline_cnn.keras -> reports/baseline_cnn_metadata.json).
        """
        stem = Path(self.checkpoint_path).stem
        meta_path = Path(self.config["paths"]["reports_dir"]) / f"{stem}_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("class_names")
        return None

    def predict(self, image_source, top_k: int | None = None) -> PredictionResult:
        """Dự đoán 1 ảnh. image_source: path str, file-like object, hoặc PIL.Image."""
        top_k = top_k or self.config["inference"]["top_k"]
        img_size = self.config["data"]["img_size"]

        array = load_image_for_inference(image_source, img_size)
        batch = np.expand_dims(array, axis=0)

        probs = self.model.predict(batch, verbose=0)[0]

        top_k = min(top_k, len(self.class_names))
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_k_results = [
            {"class": self.class_names[i], "probability": float(probs[i])}
            for i in top_indices
        ]

        return PredictionResult(
            predicted_class=self.class_names[top_indices[0]],
            confidence=float(probs[top_indices[0]]),
            top_k=top_k_results,
        )

    def model_info(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_size": self.config["data"]["img_size"],
            "num_parameters": int(self.model.count_params()),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dự đoán 1 ảnh rau/quả bằng model đã train.")
    parser.add_argument("image_path", type=str, help="Đường dẫn ảnh cần dự đoán")
    parser.add_argument("--checkpoint", type=str, default=str(Path(CONFIG["paths"]["checkpoints_dir"]) / "mobilenetv2_transfer.keras"))
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    classifier = VegetableClassifier(args.checkpoint)
    result = classifier.predict(args.image_path, top_k=args.top_k)

    print(f"Prediction: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.2%}")
    print("Top-K predictions:")
    for item in result.top_k:
        print(f"  {item['class']}: {item['probability']:.2%}")
