"""Test end-to-end cho inference pipeline: train nhanh 1 model tí hon trên dataset giả,
lưu checkpoint, rồi dùng VegetableClassifier để predict — đảm bảo toàn bộ luồng
predict() hoạt động đúng, kể cả model_info() và xử lý lỗi ảnh không hợp lệ.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.dataset import load_datasets
from src.models.baseline_cnn import build_baseline_cnn
from src.inference.predict import VegetableClassifier, InvalidImageError, PredictionResult


@pytest.fixture
def trained_tiny_model(test_config, tmp_path):
    """Train baseline CNN 1 epoch trên dataset giả, lưu checkpoint + metadata,
    trả về (checkpoint_path, class_names).
    """
    train_ds, val_ds, test_ds, info, class_to_idx = load_datasets(test_config)
    model = build_baseline_cnn(num_classes=info.num_classes, config=test_config)
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)

    ckpt_path = tmp_path / "tiny_model.keras"
    model.save(ckpt_path)

    meta_path = tmp_path / "tiny_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"class_names": info.class_names}, f)

    return str(ckpt_path), info.class_names, str(meta_path)


def test_classifier_predict_returns_valid_result(trained_tiny_model, test_config):
    ckpt_path, class_names, _ = trained_tiny_model
    classifier = VegetableClassifier(ckpt_path, class_names=class_names, config=test_config)

    dummy_img = Image.new("RGB", (100, 100), color=(255, 140, 0))
    result = classifier.predict(dummy_img, top_k=2)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class in class_names
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k) == 2
    # top_k phải sort giảm dần theo probability
    probs = [item["probability"] for item in result.top_k]
    assert probs == sorted(probs, reverse=True)


def test_classifier_model_info(trained_tiny_model, test_config):
    ckpt_path, class_names, _ = trained_tiny_model
    classifier = VegetableClassifier(ckpt_path, class_names=class_names, config=test_config)

    info = classifier.model_info()
    assert info["num_classes"] == len(class_names)
    assert info["num_parameters"] > 0


def test_classifier_rejects_invalid_image(trained_tiny_model, test_config, tmp_path):
    ckpt_path, class_names, _ = trained_tiny_model
    classifier = VegetableClassifier(ckpt_path, class_names=class_names, config=test_config)

    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"not a real jpg content")

    with pytest.raises(InvalidImageError):
        classifier.predict(str(bad_file))


def test_classifier_missing_class_names_raises(trained_tiny_model, test_config, tmp_path, monkeypatch):
    ckpt_path, _, _ = trained_tiny_model
    # Trỏ reports_dir sang thư mục rỗng để _load_class_names_from_metadata() thất bại
    cfg = dict(test_config)
    cfg["paths"] = dict(test_config["paths"])
    cfg["paths"]["reports_dir"] = str(tmp_path / "empty_reports")

    with pytest.raises(ValueError):
        VegetableClassifier(ckpt_path, class_names=None, config=cfg)
