"""Training pipeline dùng chung cho cả 2 model (baseline CNN & transfer learning).

Chạy trực tiếp:
    python -m src.training.train --model baseline
    python -m src.training.train --model transfer

Mỗi lần train sẽ:
  1. Load dataset thật từ đĩa (không giả định số class/số ảnh).
  2. Build model tương ứng.
  3. Train với callbacks chuẩn: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
  4. Lưu lại: model checkpoint tốt nhất (.keras), training history (CSV + JSON),
     và metadata (architecture, hyperparameters, training time, params...) để
     phục vụ báo cáo/so sánh model sau này — đúng yêu cầu môn học mục 4.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf

from configs.config import CONFIG, ensure_dirs
from src.data.dataset import load_datasets, compute_class_weights
from src.evaluation.plot_history import plot_training_history
from src.models.baseline_cnn import build_baseline_cnn
from src.models.transfer_model import build_transfer_model, enable_fine_tuning
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file=str(Path(CONFIG["paths"]["logs_dir"]) / "training.log"))


def get_callbacks(checkpoint_path: str, config: dict = CONFIG) -> list[tf.keras.callbacks.Callback]:
    train_cfg = config["training"]
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["reduce_lr_factor"],
            patience=train_cfg["reduce_lr_patience"],
            min_lr=train_cfg["min_lr"],
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(Path(config["paths"]["logs_dir"]) / f"{Path(checkpoint_path).stem}_history.csv")
        ),
    ]


def _save_metadata(metadata: dict, name: str, config: dict = CONFIG):
    out_path = Path(config["paths"]["reports_dir"]) / f"{name}_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu metadata training tại: {out_path}")


def train_baseline(config: dict = CONFIG) -> dict:
    """Train Model 1 — Baseline CNN. Trả về dict metadata kết quả training."""
    ensure_dirs()
    train_ds, val_ds, test_ds, info, class_to_idx = load_datasets(config)
    logger.info(f"Dataset: {info.num_classes} classes, {info.total_images} ảnh")

    model = build_baseline_cnn(num_classes=info.num_classes, config=config)
    checkpoint_path = str(Path(config["paths"]["checkpoints_dir"]) / "baseline_cnn.keras")

    class_weight = compute_class_weights(info, class_to_idx, config)
    if class_weight:
        logger.info(f"Dataset mất cân bằng -> áp dụng class_weight cho {len(class_weight)} class")
    else:
        logger.info("Không áp dụng class_weight (dataset cân bằng hoặc tính năng bị tắt)")

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["baseline_cnn"]["epochs"],
        callbacks=get_callbacks(checkpoint_path, config),
        class_weight=class_weight,
        verbose=1,
    )
    elapsed = time.time() - start

    # Vẽ biểu đồ Training vs Validation (Accuracy & Loss)
    curves_path = Path(config["paths"]["figures_dir"]) / "baseline_cnn_training_curves.png"
    plot_training_history(history.history, "Baseline CNN", str(curves_path))
    logger.info(f"Đã lưu biểu đồ training curves: {curves_path}")

    metadata = {
        "model_name": config["baseline_cnn"]["name"],
        "architecture": "Custom CNN (3x Conv2D-BN-MaxPool + GAP + Dense)",
        "num_parameters": int(model.count_params()),
        "input_size": config["data"]["img_size"],
        "batch_size": config["data"]["batch_size"],
        "epochs_configured": config["baseline_cnn"]["epochs"],
        "epochs_ran": len(history.history["loss"]),
        "optimizer": "Adam",
        "learning_rate": config["baseline_cnn"]["learning_rate"],
        "loss_function": config["training"]["loss"],
        "training_time_seconds": round(elapsed, 1),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "class_weight_applied": class_weight is not None,
        "checkpoint_path": checkpoint_path,
        "training_curves_figure": str(curves_path),
        "class_names": info.class_names,
    }
    _save_metadata(metadata, "baseline_cnn", config)
    logger.info(f"Baseline CNN training xong sau {elapsed:.1f}s. Best val_accuracy={metadata['best_val_accuracy']:.4f}")
    return metadata


def train_transfer(config: dict = CONFIG) -> dict:
    """Train Model 2 — MobileNetV2 transfer learning (2 giai đoạn). Trả về metadata."""
    ensure_dirs()
    train_ds, val_ds, test_ds, info, class_to_idx = load_datasets(config)
    logger.info(f"Dataset: {info.num_classes} classes, {info.total_images} ảnh")

    model, base_model = build_transfer_model(num_classes=info.num_classes, config=config)
    checkpoint_path = str(Path(config["paths"]["checkpoints_dir"]) / "mobilenetv2_transfer.keras")

    class_weight = compute_class_weights(info, class_to_idx, config)
    if class_weight:
        logger.info(f"Dataset mất cân bằng -> áp dụng class_weight cho {len(class_weight)} class")
    else:
        logger.info("Không áp dụng class_weight (dataset cân bằng hoặc tính năng bị tắt)")

    start = time.time()

    # --- Giai đoạn 1: feature extraction (base model đóng băng) ---
    logger.info("Giai đoạn 1/2: Feature extraction (base model bị freeze)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["transfer_model"]["epochs_feature_extraction"],
        callbacks=get_callbacks(checkpoint_path, config),
        class_weight=class_weight,
        verbose=1,
    )

    # --- Giai đoạn 2: fine-tuning (mở khóa 1 phần base model) ---
    logger.info("Giai đoạn 2/2: Fine-tuning (mở khóa các layer cuối của MobileNetV2)...")
    model = enable_fine_tuning(model, base_model, config)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["transfer_model"]["epochs_fine_tune"],
        callbacks=get_callbacks(checkpoint_path, config),
        class_weight=class_weight,
        verbose=1,
    )

    elapsed = time.time() - start

    # Gộp lịch sử 2 giai đoạn để vẽ 1 đồ thị liền mạch
    combined_history = {
        key: history1.history[key] + history2.history[key]
        for key in history1.history
    }
    hist_df = pd.DataFrame(combined_history)
    hist_path = Path(config["paths"]["logs_dir"]) / "mobilenetv2_transfer_history_combined.csv"
    hist_df.to_csv(hist_path, index=False)

    # Vẽ biểu đồ Training vs Validation, có đường phân tách 2 giai đoạn để thấy
    # rõ tác động của bước fine-tuning lên accuracy/loss.
    curves_path = Path(config["paths"]["figures_dir"]) / "mobilenetv2_transfer_training_curves.png"
    plot_training_history(
        combined_history, "MobileNetV2 (Transfer Learning)", str(curves_path),
        phase_split_epoch=len(history1.history["loss"]),
    )
    logger.info(f"Đã lưu biểu đồ training curves: {curves_path}")

    metadata = {
        "model_name": config["transfer_model"]["name"],
        "architecture": "MobileNetV2 (ImageNet pretrained) + GAP + Dense head",
        "num_parameters": int(model.count_params()),
        "trainable_parameters_stage1": int(sum(tf.size(v).numpy() for v in history1.model.trainable_variables)) if hasattr(history1, "model") else None,
        "input_size": config["data"]["img_size"],
        "batch_size": config["data"]["batch_size"],
        "epochs_feature_extraction": len(history1.history["loss"]),
        "epochs_fine_tune": len(history2.history["loss"]),
        "optimizer": "Adam",
        "learning_rate_stage1": config["transfer_model"]["learning_rate"],
        "learning_rate_stage2_finetune": config["transfer_model"]["fine_tune_learning_rate"],
        "loss_function": config["training"]["loss"],
        "training_time_seconds": round(elapsed, 1),
        "final_train_accuracy": float(combined_history["accuracy"][-1]),
        "final_val_accuracy": float(combined_history["val_accuracy"][-1]),
        "final_train_loss": float(combined_history["loss"][-1]),
        "final_val_loss": float(combined_history["val_loss"][-1]),
        "best_val_accuracy": float(max(combined_history["val_accuracy"])),
        "class_weight_applied": class_weight is not None,
        "checkpoint_path": checkpoint_path,
        "training_curves_figure": str(curves_path),
        "class_names": info.class_names,
    }
    _save_metadata(metadata, "mobilenetv2_transfer", config)
    logger.info(f"Transfer model training xong sau {elapsed:.1f}s. Best val_accuracy={metadata['best_val_accuracy']:.4f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train model phân loại rau củ quả.")
    parser.add_argument(
        "--model", choices=["baseline", "transfer", "both"], default="both",
        help="Model cần train: baseline (CNN tự xây), transfer (MobileNetV2), hoặc both (cả 2, mặc định).",
    )
    args = parser.parse_args()

    if args.model in ("baseline", "both"):
        train_baseline()
    if args.model in ("transfer", "both"):
        train_transfer()


if __name__ == "__main__":
    main()
