"""
src/features/feature_eda.py
---------------------------
NGƯỜI LÀM: Nguyen Duc Tai

Mục đích:
    Các hàm vẽ biểu đồ EDA liên quan đến ĐẶC TRƯNG (HOG, màu, PCA) — được gọi từ
    notebooks/EDA.ipynb. Tách ra file riêng (giống src/data/eda.py) để notebook
    gọn và logic có thể test lại.

    Mọi hàm vẽ đều: nhận save_path (tuỳ chọn) để lưu ảnh cho báo cáo Word,
    và trả về matplotlib Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.features.hog_features import (
    COLOR_HIST_BINS,
    DEFAULT_IMG_SIZE,
    extract_color_histogram,
    extract_hog_features,
    load_and_resize,
)


def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def sample_paths_balanced(splits_class_map: dict, n_per_class: int = 30, seed: int = 42):
    """
    Lấy mẫu CÂN BẰNG: tối đa n_per_class ảnh mỗi class (chọn ngẫu nhiên, cố định seed).

    Dùng cho EDA đặc trưng — trích HOG cho toàn bộ dataset tốn thời gian, còn mẫu
    cân bằng vẫn đủ để vẽ biểu đồ PCA/tương quan đại diện.

    Returns:
        (paths, labels): list đường dẫn ảnh, list tên class tương ứng (cùng thứ tự)
    """
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for cls in sorted(splits_class_map):
        files = splits_class_map[cls]
        k = min(n_per_class, len(files))
        idx = rng.choice(len(files), size=k, replace=False)
        paths.extend(files[i] for i in sorted(idx))
        labels.extend([cls] * k)
    return paths, labels


def plot_hog_examples(
    splits_class_map: dict,
    class_names: list,
    n_examples: int = 3,
    img_size: tuple = DEFAULT_IMG_SIZE,
    seed: int = 42,
    save_path: str | None = None,
):
    """
    Minh họa HOG: mỗi hàng là 1 class — cột trái ảnh gốc (sau resize), cột phải ảnh HOG.
    Trong ảnh HOG, các nét sáng thể hiện hướng cạnh/đường viền mà HOG "nhìn thấy".
    """
    from skimage import exposure

    rng = np.random.default_rng(seed)
    chosen = rng.choice(class_names, size=min(n_examples, len(class_names)), replace=False)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)), squeeze=False)
    for row, cls in enumerate(chosen):
        files = splits_class_map[cls]
        img = load_and_resize(files[int(rng.integers(len(files)))], img_size)
        _, hog_img = extract_hog_features(img, visualize=True)
        hog_img = exposure.rescale_intensity(hog_img, in_range=(0, np.percentile(hog_img, 99) + 1e-8))

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Ảnh gốc — {cls}", fontsize=10)
        axes[row, 1].imshow(hog_img, cmap="gray")
        axes[row, 1].set_title("Ảnh HOG", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle("Minh họa HOG: ảnh gốc và ảnh HOG", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_color_histograms(
    splits_class_map: dict,
    class_names: list,
    n_examples: int = 3,
    bins: int = COLOR_HIST_BINS,
    img_size: tuple = DEFAULT_IMG_SIZE,
    seed: int = 42,
    save_path: str | None = None,
):
    """
    Vẽ histogram màu R/G/B của vài class (mỗi class 1 ảnh) cạnh ảnh gốc.
    Cho thấy vì sao màu là đặc trưng hữu ích (ớt đỏ ≠ rau cải xanh).
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(class_names, size=min(n_examples, len(class_names)), replace=False)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 2.8 * len(chosen)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 2.2]})
    x = np.arange(bins)
    for row, cls in enumerate(chosen):
        files = splits_class_map[cls]
        img = load_and_resize(files[int(rng.integers(len(files)))], img_size)
        hist = extract_color_histogram(img, bins).reshape(3, bins)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(cls, fontsize=10)
        axes[row, 0].axis("off")
        for ch, color, name in zip(range(3), ("#D62728", "#2CA02C", "#1F77B4"), "RGB"):
            axes[row, 1].plot(x, hist[ch], color=color, label=name, linewidth=1.8)
        axes[row, 1].set_xlabel("Bin màu (0 = tối → cuối = sáng)")
        axes[row, 1].set_ylabel("Tỉ lệ pixel")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(alpha=0.3)

    fig.suptitle("Histogram màu R/G/B của một số class", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dimension_reduction(n_original: int, variance_table: dict, save_path: str | None = None):
    """
    Biểu đồ cột: số chiều TRƯỚC PCA và SAU PCA (theo từng mức phương sai giữ lại).

    Args:
        n_original: số chiều trước PCA (vd 8196)
        variance_table: dict {nhãn: (số_chiều, %phương_sai)}, vd
            {"Giữ 90%": (60, 90.0), "Giữ 95%": (120, 95.0), "200 component": (200, 97.1)}
    """
    labels = ["Trước PCA\n(HOG + màu)"] + list(variance_table.keys())
    dims = [n_original] + [v[0] for v in variance_table.values()]
    colors = ["#999999"] + ["#4C9A6B"] * len(variance_table)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, dims, color=colors)
    ax.set_yscale("log")        # thang log để cột sau PCA không bị lép so với cột 8196
    ax.set_ylabel("Số chiều đặc trưng (thang log)")
    ax.set_title("Số chiều trước và sau PCA")

    ax.text(bars[0].get_x() + bars[0].get_width() / 2, dims[0], f"{dims[0]:,}",
            ha="center", va="bottom", fontsize=10)
    for bar, (_, (d, var)) in zip(bars[1:], variance_table.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, d,
                f"{d}\n({var:.1f}% phương sai)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(top=max(dims) * 3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Ma trận tương quan giữa các cột; cột hằng số (không có phương sai) → tương quan 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def plot_color_correlation(X_color: np.ndarray, bins: int = COLOR_HIST_BINS, save_path: str | None = None):
    """
    Heatmap tương quan giữa 96 đặc trưng màu (R: 0..bins-1, G: bins..2*bins-1, B: ...).
    Các bin màu gần nhau và các kênh R/G/B thường tương quan → lý do cần PCA.
    """
    corr = correlation_matrix(X_color)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    for b in range(1, 3):                       # đường kẻ phân cách kênh R | G | B
        ax.axhline(b * bins - 0.5, color="black", linewidth=0.8)
        ax.axvline(b * bins - 0.5, color="black", linewidth=0.8)
    ticks = [bins // 2 + i * bins for i in range(3)]
    ax.set_xticks(ticks, labels=["Kênh R", "Kênh G", "Kênh B"])
    ax.set_yticks(ticks, labels=["Kênh R", "Kênh G", "Kênh B"])
    ax.set_title("Tương quan giữa các đặc trưng màu (Pearson)")
    fig.colorbar(im, ax=ax, label="Hệ số tương quan")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pca_scatter(Z: np.ndarray, labels: list, classes_to_show: list | None = None,
                     save_path: str | None = None):
    """
    Scatter 2 component đầu (PC1 vs PC2), tô màu theo class.
    Class nào tách khỏi nhóm khác → dễ phân loại; class chồng lên nhau → dễ bị nhầm.
    """
    labels = np.asarray(labels)
    classes = classes_to_show or list(dict.fromkeys(labels))[:8]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        m = labels == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.7, color=cmap(i % 10), label=cls)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Phân bố {len(classes)} class trên 2 thành phần chính đầu tiên")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig