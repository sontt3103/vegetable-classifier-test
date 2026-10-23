"""Test cho src/data/dataset.py — dùng dataset giả lập (tiny_dataset fixture)."""

import numpy as np
import pytest

from src.data.dataset import discover_dataset, load_datasets, DatasetInfo


def test_discover_dataset_finds_all_classes(test_config):
    splits, info = discover_dataset(test_config)
    assert info.num_classes == 3
    assert set(info.class_names) == {"carrot", "tomato", "potato"}
    assert info.total_images == (8 + 4 + 4) * 3


def test_discover_dataset_missing_data_raises(test_config, tmp_path):
    empty_config = dict(test_config)
    empty_config["paths"] = dict(test_config["paths"])
    empty_config["paths"]["data_dir"] = str(tmp_path / "does_not_exist")
    empty_config["paths"]["train_dir"] = str(tmp_path / "does_not_exist" / "train")
    empty_config["paths"]["val_dir"] = str(tmp_path / "does_not_exist" / "validation")
    empty_config["paths"]["test_dir"] = str(tmp_path / "does_not_exist" / "test")

    with pytest.raises(FileNotFoundError):
        discover_dataset(empty_config)


def test_load_datasets_batch_shape_and_range(test_config):
    train_ds, val_ds, test_ds, info, class_to_idx = load_datasets(test_config)

    images, labels = next(iter(train_ds))
    img_size = test_config["data"]["img_size"]

    assert images.shape[1:] == (img_size, img_size, 3)
    assert labels.shape[1] == info.num_classes

    min_val = float(np.min(images.numpy()))
    max_val = float(np.max(images.numpy()))
    assert 0.0 <= min_val <= max_val <= 1.0, "Pixel phải được normalize về [0, 1]"


def test_val_and_test_have_no_augmentation_shuffle_mismatch(test_config):
    """Đảm bảo val/test dataset không bị lẫn augmentation (chỉ train mới augment)."""
    _, val_ds, test_ds, info, _ = load_datasets(test_config)

    # Lấy cùng 1 batch 2 lần liên tiếp từ val_ds — vì val KHÔNG shuffle và KHÔNG augment,
    # 2 lần lấy phải giống hệt nhau (deterministic).
    batch1_images, _ = next(iter(val_ds))
    batch2_images, _ = next(iter(val_ds))
    np.testing.assert_array_equal(batch1_images.numpy(), batch2_images.numpy())
