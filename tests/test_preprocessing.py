"""Test cho preprocessing/augmentation — kiểm tra ảnh không bị lỗi shape/range sau augment,
và inference preprocessing xử lý đúng ảnh hợp lệ lẫn không hợp lệ.
"""

import io

import numpy as np
import pytest
from PIL import Image

from src.data.augmentation import get_augmentation_layer, preview_augmentation
from src.inference.predict import load_image_for_inference, InvalidImageError


def test_augmentation_preserves_shape_and_range(test_config):
    img_size = test_config["data"]["img_size"]
    dummy_image = np.random.rand(img_size, img_size, 3).astype("float32")

    outputs = preview_augmentation(dummy_image, n=4, config=test_config)

    assert len(outputs) == 4
    for out in outputs:
        assert out.shape == (img_size, img_size, 3)
        assert out.min() >= -0.01  # cho phép sai số nhỏ do RandomContrast/Brightness
        assert out.max() <= 1.5    # brightness/contrast có thể vượt nhẹ 1.0, không được vô lý (vd > 10)


def test_load_image_for_inference_valid_image(tmp_path):
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = tmp_path / "test.jpg"
    img.save(path)

    array = load_image_for_inference(str(path), img_size=64)
    assert array.shape == (64, 64, 3)
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_load_image_for_inference_invalid_file_raises(tmp_path):
    bad_file = tmp_path / "not_an_image.txt"
    bad_file.write_text("this is definitely not an image")

    with pytest.raises(InvalidImageError):
        load_image_for_inference(str(bad_file), img_size=64)


def test_load_image_for_inference_from_bytes():
    img = Image.new("RGB", (50, 50), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    array = load_image_for_inference(buf, img_size=32)
    assert array.shape == (32, 32, 3)
