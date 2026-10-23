"""
Centralized configuration cho toàn bộ project Vegetable Classification.

Tất cả hyperparameter và đường dẫn quan trọng được quản lý tại đây.
Không hard-code các giá trị này ở nơi khác trong code — luôn import từ đây.

Cách dùng:
    from configs.config import CONFIG
    img_size = CONFIG["data"]["img_size"]

[UPDATED]: Project đã được move sang D:\vegetable-classifier (08/18/2026)
Dataset ở: D:/vegetable-classifier/data/raw/{train, validation, test}
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn gốc của project (tự động xác định từ file location)
# Khi project ở D:\vegetable-classifier, PROJECT_ROOT = D:\vegetable-classifier
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # -----------------------------------------------------------------
    # Đường dẫn dữ liệu & output
    # -----------------------------------------------------------------
    "paths": {
        "data_dir": str(PROJECT_ROOT / "data" / "raw"),
        # Dataset Kaggle "Fruit and Vegetable Image Recognition" khi giải nén
        # thường có cấu trúc: data_dir/train, data_dir/validation, data_dir/test
        # mỗi thư mục con là 1 class. Nếu dataset của bạn có tên thư mục khác
        # (vd "val" thay vì "validation"), sửa 3 dòng dưới đây.
        "train_dir": str(PROJECT_ROOT / "data" / "raw" / "train"),
        "val_dir": str(PROJECT_ROOT / "data" / "raw" / "validation"),
        "test_dir": str(PROJECT_ROOT / "data" / "raw" / "test"),
        "checkpoints_dir": str(PROJECT_ROOT / "checkpoints"),
        "reports_dir": str(PROJECT_ROOT / "reports"),
        "figures_dir": str(PROJECT_ROOT / "reports" / "figures"),
        "logs_dir": str(PROJECT_ROOT / "reports" / "logs"),
    },

    # -----------------------------------------------------------------
    # Data / preprocessing
    # -----------------------------------------------------------------
    "data": {
        "img_size": 224,          # phù hợp với input mặc định của MobileNetV2
        "channels": 3,
        "batch_size": 32,
        "seed": 42,               # cố định seed để reproduce kết quả
        "shuffle_buffer": 1024,
        # Dataset gốc trên Kaggle đã có sẵn train/validation/test riêng biệt.
        # Cờ này chỉ dùng nếu bạn tự chuẩn bị dataset chỉ có 1 thư mục duy nhất.
        "auto_split_if_missing": True,
        "train_val_test_split": (0.7, 0.15, 0.15),
    },

    # -----------------------------------------------------------------
    # Augmentation (chỉ áp dụng cho tập train)
    # -----------------------------------------------------------------
    "augmentation": {
        "enabled": True,
        "rotation_range": 25,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "zoom_range": 0.15,
        "horizontal_flip": True,
        "brightness_range": (0.8, 1.2),
        "shear_range": 0.1,
    },

    # -----------------------------------------------------------------
    # Model — Baseline CNN (tự xây từ đầu)
    # -----------------------------------------------------------------
    "baseline_cnn": {
        "name": "baseline_cnn",
        "conv_filters": [32, 64, 128],
        "kernel_size": 3,
        "dense_units": 256,
        "dropout_rate": 0.5,
        "learning_rate": 1e-3,
        "epochs": 25,
        "optimizer": "adam",
    },

    # -----------------------------------------------------------------
    # Model — Transfer Learning (MobileNetV2)
    # -----------------------------------------------------------------
    "transfer_model": {
        "name": "mobilenetv2_transfer",
        "base_architecture": "MobileNetV2",
        "dense_units": 256,
        "dropout_rate": 0.5,
        "learning_rate": 1e-3,
        "fine_tune_learning_rate": 1e-4,  # FIXED: tăng từ 1e-5 → 1e-4 (10x lớn hơn)
        "epochs_feature_extraction": 25,  # FIXED: tăng từ 15 → 25 (head cần hội tụ kỹ)
        "epochs_fine_tune": 15,  # tăng để fine-tune tốt hơn
        "fine_tune_at_layer": 140,  # FIXED: giảm unfreezing từ 100 → 140 (chỉ 14 layers = 9%)
        "optimizer": "adam",
    },

    # -----------------------------------------------------------------
    # Training chung
    # -----------------------------------------------------------------
    "training": {
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        # Theo dõi val_accuracy (không phải val_loss): CNN có BatchNorm thường dao động
        # mạnh ở các epoch đầu, theo dõi val_loss với patience thấp sẽ dừng quá sớm.
        "early_stopping_monitor": "val_accuracy",
        "early_stopping_patience": 8,
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.5,
        "min_lr": 1e-7,
        # Tự động cân bằng trọng số giữa các class khi dataset mất cân bằng.
        # Class có ít ảnh sẽ được đánh trọng số cao hơn trong hàm loss, tránh
        # việc model "bỏ rơi" class hiếm mà vẫn đạt accuracy tổng thể cao.
        # Đặt False nếu dataset đã cân bằng (EDA sẽ cho biết điều này).
        "use_class_weight": True,
        # Chỉ áp dụng class_weight khi mức mất cân bằng vượt ngưỡng này
        # (tỉ lệ chênh lệch max-min so với trung bình). Dataset cân bằng sẵn
        # thì class_weight không có tác dụng gì, chỉ thêm phức tạp.
        "class_weight_imbalance_threshold": 0.3,
    },

    # -----------------------------------------------------------------
    # Model — Classical SVM (HOG + Color Histogram → PCA → SVM)
    # -----------------------------------------------------------------
    "svm": {
        "name": "classical_svm",
        # Giá trị C để GridSearchCV thử — C điều chỉnh độ rộng margin:
        #   C nhỏ (0.1) → margin rộng, chấp nhận phân loại sai nhiều hơn → ít overfit
        #   C lớn (100) → margin hẹp, ôm sát dữ liệu → dễ overfit
        "c_list": [0.1, 1, 10, 100],
        # Gamma điều chỉnh ảnh hưởng của từng điểm dữ liệu trong kernel RBF:
        #   'scale' → gamma = 1 / (n_features * X.var())  ← thường tốt hơn
        #   'auto'  → gamma = 1 / n_features
        "gamma_list": ["scale", "auto"],
        # Số chiều PCA: giảm từ 8196 (HOG + màu) xuống 200
        # Giữ lại ~95% phương sai, giúp SVM train nhanh hơn đáng kể
        "pca_n_components": 200,
        # Số fold cross-validation trong GridSearchCV (5 là tiêu chuẩn)
        "cv_folds": 5,
        # Kích thước ảnh đầu vào (phải khớp với hog_features.py)
        "img_size": (128, 128),
    },

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    "inference": {
        "top_k": 5,
        "confidence_threshold": 0.0,  # ngưỡng tối thiểu để coi là "chắc chắn" (0 = không lọc)
    },
}


def ensure_dirs():
    """Tạo các thư mục output nếu chưa tồn tại (idempotent, an toàn khi gọi nhiều lần)."""
    for key in ("checkpoints_dir", "reports_dir", "figures_dir", "logs_dir"):
        os.makedirs(CONFIG["paths"][key], exist_ok=True)