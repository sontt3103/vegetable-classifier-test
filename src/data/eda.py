"""Exploratory Data Analysis (EDA) cho dataset rau củ quả.

Cung cấp các hàm tính toán số liệu + vẽ biểu đồ, được gọi trực tiếp từ
notebooks/EDA.ipynb. Tách riêng logic ra khỏi notebook để:
  - dễ test (tests/test_dataset.py có thể gọi lại các hàm này)
  - tránh notebook bị phình to với code lặp lại
"""

from __future__ import annotations

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from configs.config import CONFIG
from src.data.dataset import discover_dataset, DatasetInfo


def class_distribution_df(info: DatasetInfo):
    """Trả về list[dict] (class, split, count) — dễ convert sang pandas.DataFrame."""
    rows = []
    for split, counts in info.counts_per_split.items():
        for cls, n in counts.items():
            rows.append({"class": cls, "split": split, "count": n})
    return rows


def plot_class_distribution(info: DatasetInfo, split: str = "train", save_path: str | None = None):
    """Bar chart: số lượng ảnh mỗi class trong 1 split cụ thể."""
    counts = info.counts_per_split.get(split, {})
    if not counts:
        raise ValueError(f"Không có dữ liệu cho split='{split}'")

    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(classes, values, color="#4C9A6B")
    ax.set_title(f"Phân bố số lượng ảnh theo class — tập {split}", fontsize=14)
    ax.set_xlabel("Class (loại rau/quả)")
    ax.set_ylabel("Số lượng ảnh")
    ax.tick_params(axis="x", rotation=75)

    mean_val = np.mean(values)
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1, label=f"Trung bình = {mean_val:.1f}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def detect_class_imbalance(info: DatasetInfo, split: str = "train", threshold: float = 0.3):
    """Phát hiện class imbalance đơn giản: so lệch giữa class nhiều ảnh nhất
    và ít ảnh nhất so với trung bình, vượt threshold (tỉ lệ) thì coi là mất cân bằng.

    Returns: dict tóm tắt {is_imbalanced, min_class, max_class, ratio}
    """
    counts = info.counts_per_split.get(split, {})
    if not counts:
        return {"is_imbalanced": False, "reason": "no data"}

    values = list(counts.values())
    min_cls = min(counts, key=counts.get)
    max_cls = max(counts, key=counts.get)
    mean_val = np.mean(values)
    ratio = (counts[max_cls] - counts[min_cls]) / mean_val if mean_val > 0 else 0

    return {
        "is_imbalanced": ratio > threshold,
        "min_class": min_cls,
        "min_count": counts[min_cls],
        "max_class": max_cls,
        "max_count": counts[max_cls],
        "mean_count": mean_val,
        "relative_range_ratio": ratio,
    }


def sample_images_grid(splits_class_map: dict, class_names: list[str], n_per_class: int = 1, save_path: str | None = None):
    """Vẽ lưới ảnh mẫu — mỗi class 1 (hoặc n) ảnh, để kiểm tra trực quan xem
    label có khớp nội dung ảnh không, ảnh có bị lỗi/hỏng không.
    """
    n_classes = len(class_names)
    cols = 6
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(-1)

    for i, cls in enumerate(class_names):
        ax = axes[i]
        files = splits_class_map.get(cls, [])
        if files:
            try:
                img = Image.open(files[0]).convert("RGB")
                ax.imshow(img)
            except Exception as e:  # noqa: BLE001 - hiển thị lỗi ảnh hỏng trực tiếp trên plot
                ax.text(0.5, 0.5, f"Lỗi ảnh:\n{e}", ha="center", va="center", fontsize=6)
        ax.set_title(cls, fontsize=8)
        ax.axis("off")

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Ảnh mẫu theo từng class", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def image_size_stats(splits_class_map: dict, max_sample: int = 300):
    """Lấy kích thước (width, height) của tối đa max_sample ảnh (sample ngẫu nhiên
    để không phải đọc toàn bộ dataset — vẫn đủ đại diện cho phân tích).
    """
    all_files = [f for files in splits_class_map.values() for f in files]
    rng = np.random.default_rng(CONFIG["data"]["seed"])
    sample = rng.choice(all_files, size=min(max_sample, len(all_files)), replace=False)

    sizes = []
    corrupted = []
    for f in sample:
        try:
            with Image.open(f) as img:
                sizes.append(img.size)  # (width, height)
        except Exception:  # noqa: BLE001
            corrupted.append(f)

    return sizes, corrupted


def plot_image_size_distribution(sizes: list[tuple[int, int]], save_path: str | None = None):
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(widths, bins=20, color="#4C9A6B", alpha=0.8, label="Width")
    axes[0].hist(heights, bins=20, color="#E07B39", alpha=0.6, label="Height")
    axes[0].set_title("Phân bố kích thước ảnh (pixel)")
    axes[0].set_xlabel("Pixel")
    axes[0].set_ylabel("Số lượng ảnh")
    axes[0].legend()

    axes[1].scatter(widths, heights, alpha=0.4, s=10, color="#4C9A6B")
    axes[1].set_title("Width vs Height")
    axes[1].set_xlabel("Width (px)")
    axes[1].set_ylabel("Height (px)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def full_eda_report(config: dict = CONFIG, save_figures: bool = True) -> dict:
    """Chạy toàn bộ EDA, trả về dict tóm tắt kết quả (dùng cho notebook hoặc report)."""
    from configs.config import ensure_dirs
    ensure_dirs()

    splits, info = discover_dataset(config)
    figures_dir = config["paths"]["figures_dir"]

    summary = {
        "num_classes": info.num_classes,
        "class_names": info.class_names,
        "total_images": info.total_images,
        "counts_per_split": info.counts_per_split,
        "imbalance": detect_class_imbalance(info, split="train"),
    }

    sizes, corrupted = image_size_stats(splits["train"])
    summary["image_size_sample_count"] = len(sizes)
    summary["corrupted_images_found"] = corrupted
    summary["unique_sizes"] = len(set(sizes))

    if save_figures:
        plot_class_distribution(info, "train", os.path.join(figures_dir, "class_distribution_train.png"))
        sample_images_grid(splits["train"], info.class_names, save_path=os.path.join(figures_dir, "sample_images.png"))
        plot_image_size_distribution(sizes, save_path=os.path.join(figures_dir, "image_size_distribution.png"))
        plt.close("all")

    return summary


if __name__ == "__main__":
    result = full_eda_report()
    print(f"Số class: {result['num_classes']}")
    print(f"Tổng ảnh: {result['total_images']}")
    print(f"Class imbalance: {result['imbalance']}")
    print(f"Ảnh lỗi phát hiện được (trong mẫu): {len(result['corrupted_images_found'])}")
