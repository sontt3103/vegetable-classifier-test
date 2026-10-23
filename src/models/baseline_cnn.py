"""Model 1 — Baseline CNN, tự xây từ đầu (không dùng pretrained weights).

Mục đích của model này (theo yêu cầu môn học):
  - Có kết quả tham chiếu (reference) để so sánh với transfer learning.
  - Thể hiện hiểu biết về kiến trúc CNN cơ bản: Conv2D + Pooling + Dense.
  - Là cơ sở để phân tích transfer learning có thực sự cải thiện hay không,
    thay vì mặc định chọn 1 model duy nhất ngay từ đầu.

Kiến trúc: 3 khối Conv2D -> ReLU -> MaxPool, tăng dần số filter, theo sau
là Flatten -> Dense -> Dropout -> Dense(softmax). Đây là kiến trúc CNN kinh
điển, đủ đơn giản để giải thích trong buổi thuyết trình nhưng vẫn học được
đặc trưng thị giác cơ bản (cạnh, màu sắc, hình dạng).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from configs.config import CONFIG


def build_baseline_cnn(num_classes: int, config: dict = CONFIG) -> tf.keras.Model:
    """Xây dựng và compile baseline CNN.

    Args:
        num_classes: số lớp đầu ra, lấy từ dataset thực tế (không hard-code).
        config: config dict, mặc định lấy từ configs/config.py.
    """
    cfg = config["baseline_cnn"]
    img_size = config["data"]["img_size"]
    channels = config["data"]["channels"]

    inputs = layers.Input(shape=(img_size, img_size, channels), name="input_image")
    x = inputs

    for i, n_filters in enumerate(cfg["conv_filters"]):
        x = layers.Conv2D(
            n_filters, cfg["kernel_size"], padding="same", activation="relu",
            name=f"conv{i + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool{i + 1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", name="dense_1")(x)
    x = layers.Dropout(cfg["dropout_rate"], name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=cfg["name"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=config["training"]["loss"],
        metrics=config["training"]["metrics"],
    )
    return model


if __name__ == "__main__":
    # python -m src.models.baseline_cnn -> in model summary + tổng số parameters
    model = build_baseline_cnn(num_classes=36)
    model.summary()
    total_params = model.count_params()
    print(f"\nTổng số parameters: {total_params:,}")
