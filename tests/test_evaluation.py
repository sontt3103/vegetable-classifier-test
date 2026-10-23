"""Test cho các thành phần đánh giá & xử lý class imbalance."""

import numpy as np
import pytest

from src.data.dataset import compute_class_weights, discover_dataset
from src.evaluation.metrics import compute_metrics, detect_overfitting, per_class_dataframe
from src.evaluation.confusion_matrix import top_confused_pairs
from src.evaluation.plot_history import plot_training_history


# --------------------------- class_weight ---------------------------
def test_class_weight_none_when_balanced(test_config):
    """Dataset giả lập cân bằng (mỗi class số ảnh bằng nhau) -> không cần class_weight."""
    _, info = discover_dataset(test_config)
    class_to_idx = {c: i for i, c in enumerate(info.class_names)}
    assert compute_class_weights(info, class_to_idx, test_config) is None


def test_class_weight_computed_when_imbalanced(test_config):
    """Giả lập dataset lệch mạnh -> phải sinh weight, class hiếm có weight lớn hơn."""
    _, info = discover_dataset(test_config)
    # Ép lệch: 1 class chỉ còn 1 ảnh, các class khác 30 ảnh
    classes = info.class_names
    info.counts_per_split["train"] = {classes[0]: 1}
    info.counts_per_split["train"].update({c: 30 for c in classes[1:]})

    class_to_idx = {c: i for i, c in enumerate(classes)}
    weights = compute_class_weights(info, class_to_idx, test_config)

    assert weights is not None
    rare_idx = class_to_idx[classes[0]]
    common_idx = class_to_idx[classes[1]]
    assert weights[rare_idx] > weights[common_idx], "Class hiếm phải có trọng số cao hơn"


def test_class_weight_disabled_by_config(test_config):
    cfg = dict(test_config)
    cfg["training"] = dict(test_config["training"])
    cfg["training"]["use_class_weight"] = False

    _, info = discover_dataset(cfg)
    class_to_idx = {c: i for i, c in enumerate(info.class_names)}
    assert compute_class_weights(info, class_to_idx, cfg) is None


# --------------------------- metrics ---------------------------
def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    metrics = compute_metrics(y_true, y_pred, ["a", "b", "c"])

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_compute_metrics_detects_imperfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_metrics(y_true, y_pred, ["a", "b"])

    assert metrics["accuracy"] == 0.75
    assert metrics["f1_macro"] < 1.0


def test_per_class_dataframe_sorted_by_f1_ascending():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])  # class 'b' bị sai 1
    metrics = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    df = per_class_dataframe(metrics, ["a", "b", "c"])

    f1_values = df["f1_score"].tolist()
    assert f1_values == sorted(f1_values), "Phải sort tăng dần để class yếu nhất lên đầu"


def test_detect_overfitting_statuses():
    assert detect_overfitting(0.99, 0.60)["status"] == "overfitting"
    assert detect_overfitting(0.30, 0.28)["status"] == "underfitting"
    assert detect_overfitting(0.90, 0.87)["status"] == "good_fit"


# --------------------------- confusion matrix ---------------------------
def test_top_confused_pairs_excludes_diagonal():
    cm = np.array([
        [10, 5, 0],
        [2, 8, 1],
        [0, 0, 9],
    ])
    pairs = top_confused_pairs(cm, ["a", "b", "c"], top_n=5)

    assert pairs[0]["count"] == 5
    assert pairs[0]["true_class"] == "a"
    assert pairs[0]["predicted_class"] == "b"
    # Không được chứa cặp trùng nhau (đường chéo = dự đoán đúng)
    assert all(p["true_class"] != p["predicted_class"] for p in pairs)


# --------------------------- training curves ---------------------------
def test_plot_training_history_creates_file(tmp_path):
    history = {
        "accuracy": [0.3, 0.6, 0.8],
        "val_accuracy": [0.25, 0.5, 0.7],
        "loss": [1.5, 0.9, 0.5],
        "val_loss": [1.6, 1.0, 0.65],
    }
    out = tmp_path / "curves.png"
    plot_training_history(history, "Test Model", str(out))

    assert out.exists()
    assert out.stat().st_size > 1000, "File ảnh phải có nội dung thực sự"


def test_plot_training_history_rejects_incomplete_history(tmp_path):
    incomplete = {"accuracy": [0.5], "loss": [1.0]}  # thiếu val_accuracy, val_loss
    with pytest.raises(ValueError, match="thiếu các cột bắt buộc"):
        plot_training_history(incomplete, "Bad", str(tmp_path / "x.png"))
