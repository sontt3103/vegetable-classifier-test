"""Tạo dataset giả lập nhanh để test pipeline (không cần tải Kaggle)."""

import numpy as np
from pathlib import Path
from PIL import Image

# Tạo thư mục
data_root = Path("data/raw")
classes = ["apple", "banana", "carrot", "cucumber", "tomato"]
splits = ["train", "validation", "test"]

for split in splits:
    split_dir = data_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_dir = split_dir / cls
        cls_dir.mkdir(exist_ok=True)

        # Tạo 10 ảnh giả lập cho mỗi class
        num_images = 10
        for i in range(num_images):
            # Ảnh 224x224 ngẫu nhiên
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(cls_dir / f"image_{i:03d}.jpg")

print("✓ Dataset giả lập tạo xong!")
print(f"  - {len(splits)} splits (train/validation/test)")
print(f"  - {len(classes)} classes: {', '.join(classes)}")
print(f"  - {num_images} ảnh/class/split")
print("\nChạy training:")
print("  python -m src.training.train --model both")