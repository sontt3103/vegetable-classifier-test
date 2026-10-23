"""Data augmentation cho tập train.

Dataset gốc khá nhỏ (~100 ảnh/lớp), nên augmentation mạnh giúp mô hình học
được các biến thể (xoay, lật, sáng/tối, zoom...) thay vì học thuộc lòng ảnh
train, từ đó giảm overfitting.

QUAN TRỌNG: augmentation chỉ được áp dụng cho tập train. Tập validation/test
phải giữ nguyên ảnh gốc (chỉ resize + normalize) để đánh giá phản ánh đúng
khả năng tổng quát hoá của model — tránh data leakage kiểu "đánh giá trên ảnh
đã bị biến đổi ngẫu nhiên".
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from configs.config import CONFIG


def get_augmentation_layer(config: dict = CONFIG) -> tf.keras.Sequential:
    """Trả về 1 Keras Sequential gồm các augmentation layer, áp dụng on-the-fly trên GPU/CPU.

    Dùng layers thay vì ImageDataGenerator (legacy) để tương thích tốt với
    tf.data pipeline và chạy nhanh hơn.
    """
    aug_cfg = config["augmentation"]

    aug_layers = [
        layers.RandomFlip("horizontal") if aug_cfg["horizontal_flip"] else layers.Identity(),
        layers.RandomRotation(aug_cfg["rotation_range"] / 360.0),
        layers.RandomZoom(aug_cfg["zoom_range"]),
        layers.RandomTranslation(aug_cfg["height_shift_range"], aug_cfg["width_shift_range"]),
        layers.RandomContrast(aug_cfg["shear_range"] + 0.1),
        layers.RandomBrightness(
            factor=(aug_cfg["brightness_range"][0] - 1.0, aug_cfg["brightness_range"][1] - 1.0),
            value_range=(0.0, 1.0),
        ),
    ]
    return tf.keras.Sequential(aug_layers, name="augmentation")


def preview_augmentation(image: "tf.Tensor", n: int = 6, config: dict = CONFIG):
    """Trả về danh sách n ảnh sau augmentation từ 1 ảnh gốc — dùng để visual check
    trong notebook EDA (không nên chỉ tin augmentation "chạy không lỗi", phải
    nhìn bằng mắt xem ảnh có bị biến dạng quá mức không).
    """
    aug_layer = get_augmentation_layer(config)
    batch = tf.expand_dims(image, 0)
    outputs = [aug_layer(batch, training=True)[0].numpy() for _ in range(n)]
    return outputs
