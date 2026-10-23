"""Script tải dataset "Fruit and Vegetable Image Recognition" từ Kaggle.

Chuẩn bị trước khi chạy:
    1. Tạo tài khoản Kaggle, vào Account Settings -> Create New API Token
       -> tải file kaggle.json về.
    2. Đặt kaggle.json vào ~/.kaggle/kaggle.json (Linux/Mac) hoặc
       C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows).
    3. pip install kaggle (đã có trong requirements.txt)

Chạy:
    python scripts/download_dataset.py

Script sẽ tải và giải nén dataset vào data/raw/, kết quả mong đợi là cấu trúc:
    data/raw/train/<class_name>/*.jpg
    data/raw/validation/<class_name>/*.jpg
    data/raw/test/<class_name>/*.jpg
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
KAGGLE_DATASET = "kritikseth/fruit-and-vegetable-image-recognition"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except OSError as e:
        print(
            "❌ Không tìm thấy Kaggle API credentials (kaggle.json).\n"
            "Hướng dẫn:\n"
            "  1. Đăng nhập kaggle.com -> Account -> Create New API Token\n"
            "  2. Lưu file kaggle.json vào ~/.kaggle/kaggle.json (chmod 600 trên Linux/Mac)\n"
            "  3. Chạy lại script này.\n\n"
            "Nếu không muốn dùng Kaggle API, có thể tải thủ công tại:\n"
            f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"  rồi giải nén vào: {DATA_DIR}"
        )
        sys.exit(1)
    except ImportError:
        print("❌ Chưa cài package 'kaggle'. Chạy: pip install kaggle")
        sys.exit(1)

    print(f"Đang tải dataset '{KAGGLE_DATASET}' từ Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True, quiet=False)

    print(f"✅ Tải xong. Kiểm tra cấu trúc thư mục tại: {DATA_DIR}")
    _print_tree_summary(DATA_DIR)
    _flatten_if_nested(DATA_DIR)


def _print_tree_summary(data_dir: Path, max_depth: int = 2):
    for split_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        n_classes = len([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"  {split_dir.name}/ -> {n_classes} thư mục con (class)")


def _flatten_if_nested(data_dir: Path):
    """Dataset Kaggle này đôi khi giải nén ra thêm 1 lớp thư mục lồng nhau
    (vd data/raw/Fruit and Vegetable.../train/...). Hàm này kiểm tra và cảnh báo
    nếu cần di chuyển thủ công cho khớp với configs/config.py (train_dir/val_dir/test_dir).
    """
    expected = {"train", "validation", "test"}
    top_level = {p.name for p in data_dir.iterdir() if p.is_dir()}
    if expected.issubset(top_level):
        return  # đã đúng cấu trúc mong đợi

    print(
        "\n⚠️  Cấu trúc thư mục sau khi giải nén không khớp trực tiếp với "
        "data/raw/train, data/raw/validation, data/raw/test.\n"
        f"Thư mục hiện có: {sorted(top_level)}\n"
        "Hãy kiểm tra và di chuyển/đổi tên thủ công cho khớp, hoặc sửa lại "
        "đường dẫn trong configs/config.py (mục paths.train_dir/val_dir/test_dir)."
    )


if __name__ == "__main__":
    main()
