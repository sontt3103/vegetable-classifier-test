"""Model 2 — Transfer Learning với MobileNetV2 (pretrained trên ImageNet).

Lý do chọn MobileNetV2 (đã phân tích trong roadmap của project):
  - Dataset chỉ có ~100 ảnh/lớp — quá nhỏ để train 1 CNN sâu từ đầu mà không
    overfitting nặng. MobileNetV2 đã học sẵn đặc trưng thị giác tổng quát
    (cạnh, texture, hình dạng...) từ 1.4 triệu ảnh ImageNet.
  - Nhẹ (~3.5M params so với ResNet50 ~25M), phù hợp train trên Colab free
    tier / CPU, dễ deploy trong demo Streamlit.
  - Input 224x224 khớp sẵn với pipeline đã thiết kế.

Chiến lược training 2 giai đoạn (khuyến nghị chuẩn cho transfer learning):
  1. Feature extraction: đóng băng (freeze) toàn bộ base model, chỉ train
     phần classification head mới thêm vào. Learning rate cao hơn (1e-3).
  2. Fine-tuning: mở khóa 1 số layer cuối của base model, train tiếp với
     learning rate rất nhỏ (1e-5) để tinh chỉnh đặc trưng cho phù hợp với
     rau củ quả mà không phá vỡ trọng số đã học từ ImageNet.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from configs.config import CONFIG


def build_transfer_model(num_classes: int, config: dict = CONFIG, weights: str | None = "imagenet") -> tuple[tf.keras.Model, tf.keras.Model]:
    """Xây dựng model transfer learning (giai đoạn feature extraction — base bị freeze).

    Args:
        weights: "imagenet" (mặc định, cần internet để tải pretrained weights lần đầu)
            hoặc None (random init — CHỈ dùng cho unit test / CI chạy offline, không dùng
            cho training thật vì sẽ mất hết lợi ích của transfer learning).

    Returns:
        model: model đã compile, sẵn sàng train giai đoạn 1.
        base_model: reference tới base MobileNetV2 (dùng lại khi fine-tune giai đoạn 2).
    """
    cfg = config["transfer_model"]
    img_size = config["data"]["img_size"]
    channels = config["data"]["channels"]

    base_model = MobileNetV2(
        input_shape=(img_size, img_size, channels),
        include_top=False,
        weights=weights,
    )

    inputs = layers.Input(shape=(img_size, img_size, channels), name="input_image")
    # Pipeline dataset.py đã normalize ảnh về [0, 1]; MobileNetV2 cần input
    # trong khoảng [-1, 1]. Sử dụng scale=127.5, offset=-1 để chuyển [0,1] → [-1,1]
    # trực tiếp mà không cần scale qua [0,255] (tránh precision loss).
    # MobileNetV2 với include_top=False không áp dụng preprocessing riêng,
    # nên ta phải handle normalization tại đây.
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="normalize_to_minus1_1")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", name="dense_1")(x)
    x = layers.Dropout(cfg["dropout_rate"], name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=cfg["name"])

    # ⚠️ CRITICAL: Freeze base model AFTER building functional model
    # to ensure trainable state is properly set (must be after model creation)
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=config["training"]["loss"],
        metrics=config["training"]["metrics"],
    )
    return model, base_model


def enable_fine_tuning(model: tf.keras.Model, base_model: tf.keras.Model, config: dict = CONFIG) -> tf.keras.Model:
    """Mở khóa 1 phần base model để fine-tune (giai đoạn 2).

    Chỉ unfreeze từ layer thứ `fine_tune_at_layer` trở đi — giữ nguyên các
    layer đầu (học đặc trưng tổng quát low-level như cạnh, màu) vì chúng
    hữu ích cho hầu hết bài toán ảnh, chỉ tinh chỉnh các layer sâu hơn
    (đặc trưng high-level, đặc thù cho ImageNet classes) cho phù hợp với
    rau củ quả. Recompile với learning rate rất nhỏ để tránh phá hỏng
    trọng số pretrained.
    """
    cfg = config["transfer_model"]
    base_model.trainable = True

    for layer in base_model.layers[: cfg["fine_tune_at_layer"]]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["fine_tune_learning_rate"]),
        loss=config["training"]["loss"],
        metrics=config["training"]["metrics"],
    )
    return model


if __name__ == "__main__":
    # python -m src.models.transfer_model -> in model summary + số param trainable/non-trainable
    model, base_model = build_transfer_model(num_classes=36)
    model.summary()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_trainable = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(f"\nTrainable params (giai đoạn feature extraction): {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")