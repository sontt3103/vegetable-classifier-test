"""Fixtures dùng chung cho test suite.

Vì sandbox/máy chạy test có thể KHÔNG có dataset thật (dataset cần tải từ
Kaggle, khá nặng), toàn bộ test được thiết kế để tự sinh 1 bộ dataset giả
lập nhỏ (synthetic) ngay trong thư mục tạm — vẫn kiểm tra được đúng logic
thật của pipeline (load ảnh, shape, augmentation, model build, training 1
epoch, inference...) mà không phụ thuộc vào việc dataset thật đã có sẵn
hay chưa.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from configs.config import CONFIG


@pytest.fixture
def tiny_dataset(tmp_path):
    """Tạo dataset giả gồm 3 class x (train=8, validation=4, test=4 ảnh),
    ảnh 32x32 màu ngẫu nhiên nhưng có màu chủ đạo khác nhau theo class để
    dataset "học được" (không phải nhiễu hoàn toàn), đủ cho baseline_cnn học
    trong test smoke run.
    """
    classes = ["carrot", "tomato", "potato"]
    base_colors = {"carrot": (255, 140, 0), "tomato": (200, 30, 30), "potato": (150, 110, 60)}
    root = tmp_path / "data_raw"

    counts = {"train": 8, "validation": 4, "test": 4}
    rng = np.random.default_rng(0)

    for split, n in counts.items():
        for cls in classes:
            class_dir = root / split / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            base = np.array(base_colors[cls], dtype=np.int16)
            for i in range(n):
                noise = rng.integers(-30, 30, size=(32, 32, 3))
                arr = np.clip(base + noise, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(class_dir / f"{cls}_{i}.jpg")

    return root, classes


@pytest.fixture
def test_config(tiny_dataset):
    """Config đã trỏ paths về tiny_dataset, img_size nhỏ để test chạy nhanh."""
    root, classes = tiny_dataset
    cfg = copy.deepcopy(CONFIG)
    cfg["paths"]["data_dir"] = str(root)
    cfg["paths"]["train_dir"] = str(root / "train")
    cfg["paths"]["val_dir"] = str(root / "validation")
    cfg["paths"]["test_dir"] = str(root / "test")
    cfg["data"]["img_size"] = 32
    cfg["data"]["batch_size"] = 4
    cfg["data"]["shuffle_buffer"] = 16
    cfg["baseline_cnn"]["epochs"] = 1
    cfg["transfer_model"]["epochs_feature_extraction"] = 1
    cfg["transfer_model"]["epochs_fine_tune"] = 1
    return cfg
