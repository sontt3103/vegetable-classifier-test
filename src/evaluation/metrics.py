"""Tính toán metrics đánh giá model trên test set độc lập.

Theo yêu cầu môn học: KHÔNG chỉ dùng Accuracy. Cần Precision/Recall/F1 (macro
và weighted vì có thể có class imbalance), Confusion Matrix, Classification
Report, và phân tích per-class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from configs.config import CONFIG


def get_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset):
    """Chạy inference trên toàn bộ test set, trả về (y_true, y_pred, y_prob).

    y_true, y_pred: mảng int label index.
    y_prob: mảng xác suất softmax đầy đủ (dùng cho phân tích confidence).
    """
    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_prob, axis=1)
    return y_true_idx, y_pred_idx, y_prob


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Tính đầy đủ metrics tổng quan + per-class."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
    }


def per_class_dataframe(metrics: dict, class_names: list[str]) -> pd.DataFrame:
    """Chuyển classification_report dict -> DataFrame per-class, sort theo F1 tăng dần
    (dễ nhìn class nào yếu nhất lên đầu để phân tích error).
    """
    report = metrics["classification_report"]
    rows = []
    for cls in class_names:
        if cls in report:
            rows.append({
                "class": cls,
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1_score": report[cls]["f1-score"],
                "support": report[cls]["support"],
            })
    df = pd.DataFrame(rows).sort_values("f1_score", ascending=True).reset_index(drop=True)
    return df


def detect_overfitting(train_acc: float, val_acc: float, threshold: float = 0.1) -> dict:
    """Heuristic đơn giản để phát hiện overfitting/underfitting dựa trên gap
    train vs val accuracy — dùng trong report để trả lời câu hỏi môn học.
    """
    gap = train_acc - val_acc
    if gap > threshold:
        status = "overfitting"
        explanation = (
            f"Train accuracy ({train_acc:.2%}) cao hơn val accuracy ({val_acc:.2%}) "
            f"khoảng {gap:.2%}, vượt ngưỡng {threshold:.0%} -> mô hình có dấu hiệu overfitting "
            "(học thuộc tập train, kém tổng quát hoá)."
        )
    elif train_acc < 0.5 and val_acc < 0.5:
        status = "underfitting"
        explanation = (
            f"Cả train ({train_acc:.2%}) và val accuracy ({val_acc:.2%}) đều thấp "
            "-> mô hình có dấu hiệu underfitting (chưa học được đặc trưng, có thể cần "
            "tăng epochs, giảm regularization, hoặc model chưa đủ phức tạp)."
        )
    else:
        status = "good_fit"
        explanation = (
            f"Gap giữa train ({train_acc:.2%}) và val accuracy ({val_acc:.2%}) chỉ {gap:.2%}, "
            "trong ngưỡng chấp nhận được -> mô hình tổng quát hoá tốt."
        )
    return {"status": status, "gap": float(gap), "explanation": explanation}


def save_evaluation_report(metrics: dict, model_name: str, config: dict = CONFIG):
    out_path = Path(config["paths"]["reports_dir"]) / f"{model_name}_evaluation.json"
    # Bỏ classification_report chi tiết ra 1 file riêng cho gọn (JSON top-level report to)
    to_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    to_save["classification_report"] = metrics["classification_report"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)
    return out_path
