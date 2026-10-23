"""Test cho model architecture — đảm bảo build/compile đúng, output shape khớp num_classes."""

import numpy as np

from src.models.baseline_cnn import build_baseline_cnn
from src.models.transfer_model import build_transfer_model, enable_fine_tuning


def test_baseline_cnn_output_shape(test_config):
    num_classes = 3
    model = build_baseline_cnn(num_classes=num_classes, config=test_config)

    img_size = test_config["data"]["img_size"]
    dummy_input = np.random.rand(2, img_size, img_size, 3).astype("float32")
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (2, num_classes)
    # Output phải là phân phối xác suất hợp lệ (softmax) — mỗi hàng tổng ~1
    np.testing.assert_allclose(output.sum(axis=1), np.ones(2), atol=1e-4)


def test_baseline_cnn_is_compiled(test_config):
    model = build_baseline_cnn(num_classes=3, config=test_config)
    assert model.optimizer is not None
    assert model.loss == test_config["training"]["loss"]


# weights=None trong các test dưới đây: tránh phụ thuộc mạng khi chạy test/CI
# (tải ImageNet pretrained weights cần internet). Khi train thật, LUÔN dùng
# weights="imagenet" (giá trị mặc định của build_transfer_model) để có transfer
# learning thực sự — xem src/training/train.py.

def test_transfer_model_freezes_base_by_default(test_config):
    model, base_model = build_transfer_model(num_classes=3, config=test_config, weights=None)
    assert base_model.trainable is False


def test_transfer_model_output_shape(test_config):
    num_classes = 3
    model, _ = build_transfer_model(num_classes=num_classes, config=test_config, weights=None)

    img_size = test_config["data"]["img_size"]
    dummy_input = np.random.rand(2, img_size, img_size, 3).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, num_classes)


def test_enable_fine_tuning_unfreezes_base(test_config):
    model, base_model = build_transfer_model(num_classes=3, config=test_config, weights=None)
    model = enable_fine_tuning(model, base_model, config=test_config)
    assert base_model.trainable is True
