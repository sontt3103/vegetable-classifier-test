"""Vẽ confusion matrix và tìm các cặp class hay bị nhầm lẫn nhất."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], save_path: str | None = None, normalize: bool = True):
    """Vẽ heatmap confusion matrix. normalize=True: hiển thị theo tỉ lệ % (dễ so sánh
    giữa các class có support khác nhau do class imbalance).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.4), max(8, len(class_names) * 0.4)))
    sns.heatmap(
        cm_display, annot=len(class_names) <= 20, fmt=fmt, cmap="Greens",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
        square=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    title = "Confusion Matrix" + (" (normalized)" if normalize else " (counts)")
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=75)
    plt.yticks(rotation=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, cm


def top_confused_pairs(cm: np.ndarray, class_names: list[str], top_n: int = 10):
    """Tìm top-N cặp (true_class, predicted_class) bị nhầm nhiều nhất (off-diagonal cao nhất).

    Trả về list[dict] với true, predicted, count — dùng để giải thích "class nào
    dễ bị nhầm lẫn và vì sao" trong báo cáo.
    """
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_n]
