"""Data loading cho bài toán phân loại rau củ quả.

Dataset tham khảo: Kaggle "Fruit and Vegetable Image Recognition"
(kritikseth/fruit-and-vegetable-image-recognition), có sẵn 3 thư mục
train/validation/test, mỗi thư mục con là 1 class.

Module này KHÔNG giả định số lượng class hay số ảnh cụ thể — mọi thông tin
được đọc trực tiếp từ cấu trúc thư mục thực tế tại thời điểm chạy, đúng theo
nguyên tắc "không giả định dataset trước khi kiểm tra thực tế" của project.

Thiết kế:
    - Nếu tồn tại đủ 3 thư mục train/validation/test -> dùng trực tiếp
      (đây là cách "đúng" để tránh data leakage vì Kaggle đã tách sẵn theo ảnh
      chứ không phải theo phần trăm ngẫu nhiên).
    - Nếu chỉ có 1 thư mục ảnh duy nhất -> tự động chia theo tỉ lệ trong
      config (train_val_test_split), stratify theo class, với seed cố định
      để reproduce được.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from configs.config import CONFIG


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class DatasetInfo:
    """Thông tin tổng quan về dataset, dùng cho EDA và sanity-check."""

    class_names: list[str] = field(default_factory=list)
    num_classes: int = 0
    counts_per_split: dict[str, dict[str, int]] = field(default_factory=dict)
    total_images: int = 0


def _list_images_per_class(directory: str) -> dict[str, list[str]]:
    """Quét 1 thư mục dạng <directory>/<class_name>/*.jpg -> dict class -> file paths."""
    directory = Path(directory)
    result: dict[str, list[str]] = {}
    if not directory.exists():
        return result
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        files = [
            str(f) for f in sorted(class_dir.iterdir())
            if f.suffix.lower() in IMG_EXTENSIONS
        ]
        if files:
            result[class_dir.name] = files
    return result


def discover_dataset(config: dict = CONFIG) -> tuple[dict[str, dict[str, list[str]]], DatasetInfo]:
    """Kiểm tra cấu trúc dataset thực tế trên đĩa và trả về danh sách file theo split.

    Returns:
        splits: {"train": {class: [files]}, "validation": {...}, "test": {...}}
        info: DatasetInfo tổng hợp (số class, số ảnh mỗi split...)

    Raises:
        FileNotFoundError: nếu không tìm thấy dữ liệu nào cả (data_dir rỗng) —
        thông báo rõ ràng để người dùng biết cần tải dataset trước.
    """
    paths = config["paths"]
    train_dir, val_dir, test_dir = paths["train_dir"], paths["val_dir"], paths["test_dir"]

    train_map = _list_images_per_class(train_dir)
    val_map = _list_images_per_class(val_dir)
    test_map = _list_images_per_class(test_dir)

    if train_map and val_map and test_map:
        splits = {"train": train_map, "validation": val_map, "test": test_map}
    elif train_map and not val_map and not test_map:
        # Chỉ có 1 thư mục -> tự chia train/val/test theo config, stratify theo class
        if not config["data"]["auto_split_if_missing"]:
            raise FileNotFoundError(
                "Chỉ tìm thấy thư mục 'train' nhưng auto_split_if_missing=False. "
                "Hãy chuẩn bị đủ train/validation/test hoặc bật cờ auto split trong config."
            )
        splits = _auto_split(train_map, config["data"]["train_val_test_split"], config["data"]["seed"])
    else:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong '{paths['data_dir']}'. "
            "Hãy tải dataset trước (xem scripts/download_dataset.py hoặc TRAINING_GUIDE.md) "
            "và đảm bảo cấu trúc thư mục: data/raw/train/<class>/*.jpg, "
            "data/raw/validation/<class>/*.jpg, data/raw/test/<class>/*.jpg"
        )

    all_classes = sorted(set().union(*[set(m.keys()) for m in splits.values()]))
    counts = {
        split: {cls: len(files) for cls, files in class_map.items()}
        for split, class_map in splits.items()
    }
    total = sum(sum(c.values()) for c in counts.values())

    info = DatasetInfo(
        class_names=all_classes,
        num_classes=len(all_classes),
        counts_per_split=counts,
        total_images=total,
    )
    return splits, info


def _auto_split(
    class_map: dict[str, list[str]], ratios: tuple[float, float, float], seed: int
) -> dict[str, dict[str, list[str]]]:
    """Chia ảnh của mỗi class thành train/val/test theo tỉ lệ, stratified, seed cố định."""
    rng = np.random.default_rng(seed)
    train_ratio, val_ratio, _ = ratios
    splits = {"train": {}, "validation": {}, "test": {}}

    for cls, files in class_map.items():
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        splits["train"][cls] = files[:n_train]
        splits["validation"][cls] = files[n_train:n_train + n_val]
        splits["test"][cls] = files[n_train + n_val:]

    return splits


def _flatten(class_map: dict[str, list[str]], class_to_idx: dict[str, int]) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for cls, files in class_map.items():
        paths.extend(files)
        labels.extend([class_to_idx[cls]] * len(files))
    return paths, labels


def _decode_and_resize(path: str, label: int, img_size: int, num_classes: int):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32) / 255.0  # normalize về [0, 1]
    label = tf.one_hot(label, num_classes)
    return image, label


def build_tf_dataset(
    class_map: dict[str, list[str]],
    class_to_idx: dict[str, int],
    config: dict = CONFIG,
    training: bool = False,
) -> tf.data.Dataset:
    """Tạo tf.data.Dataset từ mapping class -> danh sách file.

    training=True: shuffle + augmentation (nếu bật trong config).
    training=False: KHÔNG shuffle, KHÔNG augmentation — dùng cho val/test để
    đảm bảo đánh giá công bằng, không rò rỉ augmentation vào tập đánh giá.
    """
    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]
    num_classes = len(class_to_idx)

    paths, labels = _flatten(class_map, class_to_idx)
    if len(paths) == 0:
        raise ValueError("Danh sách ảnh rỗng — không thể tạo tf.data.Dataset.")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            min(config["data"]["shuffle_buffer"], len(paths)),
            seed=config["data"]["seed"],
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, l: _decode_and_resize(p, l, img_size, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training and config["augmentation"]["enabled"]:
        from src.data.augmentation import get_augmentation_layer
        aug_layer = get_augmentation_layer(config)
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(info: DatasetInfo, class_to_idx: dict[str, int], config: dict = CONFIG) -> dict[int, float] | None:
    """Tính class_weight để xử lý class imbalance khi train.

    Công thức (giống sklearn 'balanced'):
        weight[c] = tổng_số_ảnh / (số_class * số_ảnh_của_class_c)

    Class ít ảnh -> weight > 1 (được "chú ý" nhiều hơn trong hàm loss).
    Class nhiều ảnh -> weight < 1.

    Trả về None nếu:
      - config tắt tính năng này (use_class_weight=False), HOẶC
      - dataset đã cân bằng sẵn (mức lệch dưới ngưỡng) — khi đó class_weight
        không mang lại lợi ích gì, chỉ làm training phức tạp thêm.
    """
    train_cfg = config["training"]
    if not train_cfg.get("use_class_weight", False):
        return None

    counts = info.counts_per_split.get("train", {})
    if not counts:
        return None

    values = list(counts.values())
    mean_count = sum(values) / len(values)
    if mean_count == 0:
        return None

    imbalance_ratio = (max(values) - min(values)) / mean_count
    if imbalance_ratio <= train_cfg.get("class_weight_imbalance_threshold", 0.3):
        return None  # dataset đủ cân bằng, không cần can thiệp

    total = sum(values)
    n_classes = len(counts)
    return {
        class_to_idx[cls]: total / (n_classes * n)
        for cls, n in counts.items()
        if n > 0
    }


def load_datasets(config: dict = CONFIG):
    """High-level entrypoint: khám phá dataset trên đĩa + tạo 3 tf.data.Dataset.

    Returns:
        train_ds, val_ds, test_ds, info (DatasetInfo), class_to_idx (dict)
    """
    splits, info = discover_dataset(config)
    class_to_idx = {cls: i for i, cls in enumerate(info.class_names)}

    train_ds = build_tf_dataset(splits["train"], class_to_idx, config, training=True)
    val_ds = build_tf_dataset(splits["validation"], class_to_idx, config, training=False)
    test_ds = build_tf_dataset(splits["test"], class_to_idx, config, training=False)

    return train_ds, val_ds, test_ds, info, class_to_idx


if __name__ == "__main__":
    # python -m src.data.dataset  -> sanity check nhanh, in shape của 1 batch
    train_ds, val_ds, test_ds, info, class_to_idx = load_datasets()
    print(f"Số class: {info.num_classes}")
    print(f"Tổng số ảnh: {info.total_images}")
    for split, counts in info.counts_per_split.items():
        print(f"  {split}: {sum(counts.values())} ảnh")

    for images, labels in train_ds.take(1):
        print(f"Batch shape: images={images.shape}, labels={labels.shape}")
        print(f"Giá trị pixel: min={float(tf.reduce_min(images)):.3f}, max={float(tf.reduce_max(images)):.3f}")
