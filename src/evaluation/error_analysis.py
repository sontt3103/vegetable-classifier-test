"""Phân tích các trường hợp model dự đoán sai trên test set.

Yêu cầu môn học: "phải kiểm tra các trường hợp model dự đoán sai" và hiển thị
correct/incorrect predictions kèm confidence score.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def collect_misclassified(test_ds: tf.data.Dataset, model: tf.keras.Model, class_names: list[str], max_items: int = 200):
    """Duyệt qua test set, thu thập các ảnh bị dự đoán sai kèm ảnh gốc, label thật,
    label dự đoán, và confidence — để hiển thị trực quan.

    Trả về list[dict]: {image (np.array), true_label, pred_label, confidence}
    """
    results = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        true_idx = np.argmax(labels.numpy(), axis=1)

        wrong_mask = pred_idx != true_idx
        for i in np.where(wrong_mask)[0]:
            results.append({
                "image": images[i].numpy(),
                "true_label": class_names[true_idx[i]],
                "pred_label": class_names[pred_idx[i]],
                "confidence": float(probs[i][pred_idx[i]]),
            })
            if len(results) >= max_items:
                return results
    return results


def plot_misclassified_grid(misclassified: list[dict], n: int = 12, save_path: str | None = None):
    """Vẽ lưới ảnh bị đoán sai kèm nhãn thật/dự đoán/confidence — trực quan hoá cho
    phần "Error Analysis" của báo cáo.
    """
    n = min(n, len(misclassified))
    if n == 0:
        return None

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        item = misclassified[i]
        ax = axes[i]
        ax.imshow(item["image"])
        ax.set_title(
            f"Thật: {item['true_label']}\nDự đoán: {item['pred_label']} ({item['confidence']:.1%})",
            fontsize=8, color="crimson",
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Các trường hợp dự đoán SAI trên tập test", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_correct_predictions_grid(test_ds: tf.data.Dataset, model: tf.keras.Model, class_names: list[str], n: int = 12, save_path: str | None = None):
    """Ngược lại với trên: lấy các dự đoán ĐÚNG để đối chiếu (theo yêu cầu môn học
    hiển thị cả correct và incorrect predictions)."""
    collected = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        true_idx = np.argmax(labels.numpy(), axis=1)
        correct_mask = pred_idx == true_idx
        for i in np.where(correct_mask)[0]:
            collected.append({
                "image": images[i].numpy(),
                "true_label": class_names[true_idx[i]],
                "pred_label": class_names[pred_idx[i]],
                "confidence": float(probs[i][pred_idx[i]]),
            })
            if len(collected) >= n:
                break
        if len(collected) >= n:
            break

    if not collected:
        return None

    cols = 4
    rows = int(np.ceil(len(collected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for i, item in enumerate(collected):
        axes[i].imshow(item["image"])
        axes[i].set_title(f"{item['pred_label']} ({item['confidence']:.1%})", fontsize=8, color="green")
        axes[i].axis("off")
    for j in range(len(collected), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Các trường hợp dự đoán ĐÚNG trên tập test", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
