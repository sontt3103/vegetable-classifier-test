"""
scripts/resize_dataset.py
-------------------------
NGƯỜI LÀM: TN 3

Vấn đề: ảnh gốc của dataset rất nặng (nhiều ảnh 2-8 MB, vài nghìn pixel mỗi cạnh),
trong khi CNN/MobileNetV2 chỉ dùng 224x224 và SVM chỉ dùng 128x128. Mỗi epoch máy
phải giải nén lại ảnh lớn -> train CNN rất chậm (~4 phút/epoch trên CPU).

Script này thu nhỏ MỘT LẦN mọi ảnh sao cho cạnh ngắn = 256 px (giữ tỉ lệ, đủ lớn
cho 224x224), xoay đúng chiều theo EXIF. Model nhìn thấy gần như cùng nội dung,
nhưng đọc ảnh nhanh hơn rất nhiều.

An toàn: thư mục ảnh gốc được ĐỔI TÊN thành data/raw_fullres (không xoá),
ảnh thu nhỏ ghi vào data/raw với cùng tên file -> code không cần sửa gì.
Cache đặc trưng SVM cũ (checkpoints/features) bị xoá để lần chạy sau tính lại.

Chạy:
    python scripts/resize_dataset.py
"""

from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FULL = ROOT / "data" / "raw_fullres"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SHORT_SIDE = 256


def resize_one(args):
    src, dst = args
    from PIL import Image, ImageOps
    try:
        with Image.open(src) as im:
            if im.format == "JPEG":
                im.draft("RGB", (SHORT_SIDE * 2, SHORT_SIDE * 2))  # giải nén nhanh ở độ phân giải thấp
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            scale = SHORT_SIDE / min(w, h)
            if scale < 1:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fmt = "PNG" if dst.suffix.lower() == ".png" else "JPEG"
            im.save(dst, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
        return None
    except Exception as e:  # noqa: BLE001
        return f"{src}: {e}"


def main():
    if FULL.exists():
        print(f"Đã có {FULL} -> tạo lại {RAW} từ ảnh gốc")
        if RAW.exists():
            shutil.rmtree(RAW)
    else:
        if not RAW.exists():
            sys.exit(f"Không thấy {RAW}")
        RAW.rename(FULL)
        print("Đã đổi tên: data/raw -> data/raw_fullres (ảnh gốc)")

    jobs = [(f, RAW / f.relative_to(FULL)) for f in sorted(FULL.rglob("*"))
            if f.is_file() and f.suffix.lower() in EXTS]
    before = sum(f.stat().st_size for f, _ in jobs)
    print(f"Thu nhỏ {len(jobs)} ảnh (cạnh ngắn {SHORT_SIDE}px)...")
    t0 = time.time()
    errors = []
    with ProcessPoolExecutor() as ex:
        for i, err in enumerate(ex.map(resize_one, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")
    after = sum(d.stat().st_size for _, d in jobs if d.exists())

    cache = ROOT / "checkpoints" / "features"
    if cache.exists():
        shutil.rmtree(cache)
        print("Đã xoá cache đặc trưng SVM cũ (checkpoints/features) -> chạy lại train_svm để tính lại")

    print(f"\n✓ Xong trong {time.time() - t0:.0f}s")
    print(f"  Dung lượng: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")
    if errors:
        print(f"  [!] {len(errors)} ảnh lỗi, ví dụ: {errors[0]}")
    print(f"  Ảnh gốc giữ ở: {FULL}")


if __name__ == "__main__":
    main()
