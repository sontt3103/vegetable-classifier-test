"""Entry point đánh giá 1 model đã train trên test set độc lập, và (tuỳ chọn)
so sánh 2 model với nhau để chọn final model.

Chạy:
    python -m src.evaluation.evaluate --model checkpoints/baseline_cnn.keras
    python -m src.evaluation.evaluate --compare  # so sánh cả baseline & transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from configs.config import CONFIG, ensure_dirs
from src.data.dataset import load_datasets
from src.evaluation.confusion_matrix import plot_confusion_matrix, top_confused_pairs
from src.evaluation.error_analysis import collect_misclassified, plot_misclassified_grid, plot_correct_predictions_grid
from src.evaluation.metrics import (
    compute_metrics,
    detect_overfitting,
    get_predictions,
    per_class_dataframe,
    save_evaluation_report,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(checkpoint_path: str, model_name: str, metadata_path: str | None = None, config: dict = CONFIG) -> dict:
    """Đánh giá đầy đủ 1 model: metrics, confusion matrix, error analysis.

    Returns: dict kết quả tóm tắt (dùng để so sánh model).
    """
    ensure_dirs()
    figures_dir = config["paths"]["figures_dir"]

    _, _, test_ds, info, class_to_idx = load_datasets(config)

    # Load model từ checkpoint
    # Rebuild model và load weights để tránh vấn đề serialize Lambda layers
    logger.info(f"Loading model: {model_name}")

    if "baseline" in model_name.lower():
        from src.models.baseline_cnn import build_baseline_cnn
        model = build_baseline_cnn(num_classes=info.num_classes, config=config)
        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            logger.warning(f"Could not load weights directly: {e}")
            # Fallback: cố gắng load toàn bộ model
            tf.keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(checkpoint_path)
    elif "transfer" in model_name.lower() or "mobilenet" in model_name.lower():
        from src.models.transfer_model import build_transfer_model
        result = build_transfer_model(num_classes=info.num_classes, config=config)
        # build_transfer_model trả về tuple (model, base_model)
        if isinstance(result, tuple):
            model, _ = result
        else:
            model = result
        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            logger.warning(f"Could not load weights directly: {e}")
            # Fallback: cố gắng load toàn bộ model
            tf.keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    logger.info(f"Đang chạy inference trên test set ({info.num_classes} classes)...")
    y_true, y_pred, y_prob = get_predictions(model, test_ds)

    metrics = compute_metrics(y_true, y_pred, info.class_names)
    save_evaluation_report(metrics, model_name, config)

    df = per_class_dataframe(metrics, info.class_names)
    df.to_csv(Path(config["paths"]["reports_dir"]) / f"{model_name}_per_class_metrics.csv", index=False)

    fig_cm, cm = plot_confusion_matrix(
        y_true, y_pred, info.class_names,
        save_path=str(Path(figures_dir) / f"{model_name}_confusion_matrix.png"),
    )
    confused_pairs = top_confused_pairs(cm, info.class_names)

    misclassified = collect_misclassified(test_ds, model, info.class_names)
    plot_misclassified_grid(misclassified, save_path=str(Path(figures_dir) / f"{model_name}_misclassified.png"))
    plot_correct_predictions_grid(test_ds, model, info.class_names, save_path=str(Path(figures_dir) / f"{model_name}_correct_predictions.png"))

    overfit_info = None
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, encoding="utf-8") as f:
            train_meta = json.load(f)
        overfit_info = detect_overfitting(train_meta["final_train_accuracy"], train_meta["final_val_accuracy"])

    summary = {
        "model_name": model_name,
        "test_accuracy": metrics["accuracy"],
        "test_precision_macro": metrics["precision_macro"],
        "test_recall_macro": metrics["recall_macro"],
        "test_f1_macro": metrics["f1_macro"],
        "test_f1_weighted": metrics["f1_weighted"],
        "num_test_samples": int(len(y_true)),
        "best_class": df.iloc[-1]["class"] if len(df) else None,
        "worst_class": df.iloc[0]["class"] if len(df) else None,
        "top_confused_pairs": confused_pairs[:5],
        "num_misclassified": len(misclassified),
        "overfitting_analysis": overfit_info,
    }

    summary_path = Path(config["paths"]["reports_dir"]) / f"{model_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"[{model_name}] Test accuracy: {metrics['accuracy']:.2%} | Macro F1: {metrics['f1_macro']:.3f}")
    logger.info(f"[{model_name}] Class tốt nhất: {summary['best_class']} | Class yếu nhất: {summary['worst_class']}")
    return summary


def compare_models(config: dict = CONFIG) -> pd.DataFrame:
    """So sánh baseline CNN vs MobileNetV2 transfer, tạo bảng so sánh + lưu CSV."""
    reports_dir = Path(config["paths"]["reports_dir"])
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])

    results = []
    for name, ckpt_file in [
        ("baseline_cnn", "baseline_cnn.keras"),
        ("mobilenetv2_transfer", "mobilenetv2_transfer.keras"),
    ]:
        ckpt_path = checkpoints_dir / ckpt_file
        meta_path = reports_dir / f"{name}_metadata.json"
        if not ckpt_path.exists():
            logger.warning(f"Bỏ qua {name}: chưa tìm thấy checkpoint tại {ckpt_path}")
            continue

        summary = evaluate_model(str(ckpt_path), name, str(meta_path), config)
        with open(meta_path, encoding="utf-8") as f:
            train_meta = json.load(f)

        results.append({
            "Model": name,
            "Test Accuracy": summary["test_accuracy"],
            "Precision (macro)": summary["test_precision_macro"],
            "Recall (macro)": summary["test_recall_macro"],
            "F1-score (macro)": summary["test_f1_macro"],
            "Training Time (s)": train_meta.get("training_time_seconds"),
            "Num Parameters": train_meta.get("num_parameters"),
        })

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Sinh báo cáo Markdown tổng hợp để dùng trực tiếp khi viết báo cáo môn học
    from src.evaluation.report import generate_report
    report_path = generate_report(config)
    logger.info(f"Đã sinh báo cáo tổng hợp: {report_path}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model trên test set độc lập.")
    parser.add_argument("--model", type=str, help="Đường dẫn checkpoint .keras hoặc tên model (baseline/transfer)")
    parser.add_argument("--name", type=str, default=None, help="Tên model (mặc định lấy từ tên file)")
    parser.add_argument("--compare", action="store_true", help="So sánh cả baseline_cnn và mobilenetv2_transfer")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    elif args.model:
        # Map model names to checkpoint paths
        model_mapping = {
            "baseline": "baseline_cnn.keras",
            "transfer": "mobilenetv2_transfer.keras",
            "mobilenet": "mobilenetv2_transfer.keras",
        }

        # Nếu arg là model name, convert thành full path; nếu đã là path thì dùng trực tiếp
        checkpoints_dir = Path(CONFIG["paths"]["checkpoints_dir"])
        if args.model.lower() in model_mapping:
            checkpoint_path = str(checkpoints_dir / model_mapping[args.model.lower()])
            name = args.name or args.model.lower()
        else:
            checkpoint_path = args.model
            name = args.name or Path(args.model).stem

        evaluate_model(checkpoint_path, name)
    else:
        parser.error("Cần chỉ định --model <checkpoint|baseline|transfer> hoặc --compare")


if __name__ == "__main__":
    main()