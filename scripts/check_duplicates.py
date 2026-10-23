"""
scripts/check_duplicates.py
---------------------------
NGƯỜI LÀM: TN 3

Kiểm tra chất lượng data/raw/{train,validation,test} TRƯỚC KHI TRAIN:
    1. Ảnh NHIỄU NGẪU NHIÊN (do create_test_dataset.py sinh ra, không phải ảnh thật)
    2. Ảnh test trùng với ảnh train  -> rò rỉ dữ liệu, kết quả test "đẹp" giả
    3. Ảnh validation trùng ảnh test -> val accuracy không độc lập
    4. Cùng 1 ảnh nằm ở NHIỀU class   -> nhãn sai
    5. Ảnh hỏng / không đọc được

Hash dùng:
    - md5        : trùng y hệt từng byte
    - ahash 16x16: nghi trùng gần đúng (resize/nén lại) -> cần mở ảnh ra xem

Chạy:
    python scripts/check_duplicates.py               # chỉ kiểm tra, KHÔNG sửa gì
    python scripts/check_duplicates.py --move-noise  # chuyển ảnh nhiễu sang data/_removed/
Output:
    reports/duplicate_check.csv
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
REMOVED = ROOT / "data" / "_removed"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_gray(path: Path):
    """Đọc ảnh xám, chạy được với đường dẫn có dấu tiếng Việt trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            from PIL import Image
            with Image.open(path) as im:
                img = np.array(im.convert("L"))
        except Exception:
            return None
    return img


def ahash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    return np.packbits((small > small.mean()).flatten()).tobytes().hex()


def is_random_noise(gray: np.ndarray) -> bool:
    """Ảnh thật: pixel cạnh nhau rất giống nhau -> tương quan ~0.8-0.99.
    Ảnh nhiễu ngẫu nhiên (create_test_dataset.py): pixel độc lập -> tương quan ~0."""
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    a, b = g[:, :-1].ravel(), g[:, 1:].ravel()
    if a.std() < 1e-3 or b.std() < 1e-3:
        return False  # ảnh một màu, không phải nhiễu
    return float(np.corrcoef(a, b)[0, 1]) < 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move-noise", action="store_true",
                    help="Chuyển ảnh nhiễu ngẫu nhiên sang data/_removed/ (giữ cấu trúc thư mục)")
    args = ap.parse_args()

    records = []
    for split in ("train", "validation", "test"):
        sdir = DATA / split
        if not sdir.exists():
            continue
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in EXTS:
                    records.append((split, cdir.name, f))
    if not records:
        sys.exit(f"Không thấy ảnh trong {DATA}")
    print(f"Đang kiểm tra {len(records)} ảnh...")

    groups = {"md5": defaultdict(list), "ahash": defaultdict(list)}
    unreadable, noise = [], []
    for i, (split, cls, f) in enumerate(records):
        groups["md5"][hashlib.md5(f.read_bytes()).hexdigest()].append((split, cls, f))
        g = read_gray(f)
        if g is None:
            unreadable.append(f)
        else:
            groups["ahash"][ahash(g)].append((split, cls, f))
            if is_random_noise(g):
                noise.append((split, cls, f))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(records)}")

    rows = []
    for kind, g in groups.items():
        for items in g.values():
            if len(items) < 2:
                continue
            splits = sorted({s for s, _, _ in items})
            classes = sorted({c for _, c, _ in items})
            rows.append({
                "issue": f"duplicate_{kind}", "n_files": len(items),
                "splits": ",".join(splits), "classes": ",".join(classes),
                "cross_split": len(splits) > 1, "cross_class": len(classes) > 1,
                "files": " | ".join(str(f.relative_to(DATA)) for _, _, f in items),
            })
    for s, c, f in noise:
        rows.append({"issue": "random_noise", "n_files": 1, "splits": s, "classes": c,
                     "cross_split": False, "cross_class": False, "files": str(f.relative_to(DATA))})
    for f in unreadable:
        rows.append({"issue": "unreadable", "n_files": 1, "splits": "", "classes": "",
                     "cross_split": False, "cross_class": False, "files": str(f.relative_to(DATA))})

    df = pd.DataFrame(rows, columns=["issue", "n_files", "splits", "classes",
                                     "cross_split", "cross_class", "files"])
    out = ROOT / "reports" / "duplicate_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # ---- Tóm tắt ----
    counts = defaultdict(lambda: defaultdict(int))
    for s, _, _ in records:
        counts[s]["total"] += 1
    print("\n===== SỐ ẢNH =====")
    for s in ("train", "validation", "test"):
        if s in counts:
            print(f"  {s:<11}: {counts[s]['total']}")

    print("\n===== ẢNH NHIỄU NGẪU NHIÊN (không phải ảnh rau củ) =====")
    if noise:
        by = defaultdict(int)
        for s, c, _ in noise:
            by[(s, c)] += 1
        for (s, c), n in sorted(by.items()):
            print(f"  {s}/{c}: {n} ảnh")
        print(f"  TỔNG: {len(noise)} ảnh  -> PHẢI loại bỏ trước khi train "
              "(chạy lại với --move-noise)")
    else:
        print("  Không có. ✓")

    print(f"\n===== ẢNH HỎNG: {len(unreadable)} =====")
    for f in unreadable[:5]:
        print("  ", f.relative_to(DATA))

    for kind, label in (("md5", "TRÙNG Y HỆT (md5)"), ("ahash", "NGHI TRÙNG GẦN ĐÚNG (ahash, cần mở ra xem)")):
        d = df[df.issue == f"duplicate_{kind}"]
        has = lambda col, a, b: (d[col].str.contains(a) & d[col].str.contains(b)).sum()
        print(f"\n===== {label}: {len(d)} nhóm =====")
        print(f"  train ∩ test      : {has('splits', 'train', 'test')} nhóm")
        print(f"  validation ∩ test : {has('splits', 'validation', 'test')} nhóm")
        print(f"  khác class        : {int(d.cross_class.sum())} nhóm")
    print(f"\nChi tiết: {out}")

    if args.move_noise and noise:
        for s, c, f in noise:
            dst = REMOVED / s / c / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
        print(f"\n✓ Đã chuyển {len(noise)} ảnh nhiễu sang {REMOVED} (xoá thư mục này nếu chắc chắn).")


if __name__ == "__main__":
    main()
