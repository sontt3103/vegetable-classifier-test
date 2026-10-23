"""
scripts/make_clean_split.py
---------------------------
NGƯỜI LÀM: TN 3

Vấn đề của dataset Kaggle gốc (kết quả scripts/check_duplicates.py):
    - ~89% ảnh test có y hệt trong train  -> rò rỉ dữ liệu, test accuracy cao giả
    - validation và test gần như cùng 1 bộ ảnh
    - một số ảnh nằm ở 2 class khác nhau (nhãn mâu thuẫn), có ảnh nhiễu

Script này tạo bộ dữ liệu SẠCH:
    1. Gom ảnh của cả 3 split, nhóm các ảnh trùng (md5 + ahash)
    2. Mỗi nhóm trùng chỉ giữ 1 ảnh
    3. Bỏ nhóm ảnh nằm ở >1 class (không biết nhãn đúng) và ảnh nhiễu ngẫu nhiên
    4. Chia lại train/validation/test theo tỉ lệ trong config (mặc định 70/15/15),
       stratified theo class, seed cố định 42 -> chạy lại ra đúng kết quả cũ

An toàn: dataset gốc được ĐỔI TÊN thành data/raw_kaggle (không xoá),
bộ sạch được ghi vào data/raw -> toàn bộ code train/evaluate không cần sửa gì.

Chạy:
    python scripts/make_clean_split.py
Kết quả:
    data/raw_kaggle/   (dataset gốc, giữ nguyên)
    data/raw/          (dataset sạch, chia lại)
    reports/clean_split_summary.csv, reports/clean_split_removed.csv
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
RAW = DATA / "raw"
ORIG = DATA / "raw_kaggle"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "validation", "test")


def read_gray(path: Path):
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


def ahash(gray):
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    return np.packbits((small > small.mean()).flatten()).tobytes().hex()


def is_random_noise(gray):
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    a, b = g[:, :-1].ravel(), g[:, 1:].ravel()
    if a.std() < 1e-3 or b.std() < 1e-3:
        return False
    return float(np.corrcoef(a, b)[0, 1]) < 0.3


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    try:
        from configs.config import CONFIG
        ratios = CONFIG["data"]["train_val_test_split"]
        seed = CONFIG["data"]["seed"]
    except Exception:
        ratios, seed = (0.7, 0.15, 0.15), 42

    # ---- 1. Chuẩn bị: giữ dataset gốc ở data/raw_kaggle ----
    if ORIG.exists():
        print(f"Đã có {ORIG} (dataset gốc) -> dùng làm nguồn, tạo lại {RAW}")
        if RAW.exists():
            shutil.rmtree(RAW)
    else:
        if not all((RAW / s).is_dir() for s in SPLITS):
            sys.exit(f"Không thấy {RAW}/train, validation, test. Kiểm tra lại bước tải dataset.")
        RAW.rename(ORIG)
        print(f"Đã đổi tên dataset gốc: data/raw -> data/raw_kaggle")

    # ---- 2. Đọc toàn bộ ảnh ----
    items = []  # (split, class, path)
    for s in SPLITS:
        for cdir in sorted(p for p in (ORIG / s).iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in EXTS:
                    items.append((s, cdir.name, f))
    classes = sorted({c for _, c, _ in items})
    print(f"Ảnh gốc: {len(items)} | {len(classes)} class")

    uf = UnionFind(len(items))
    first_by_key = {}
    noise, unreadable = set(), set()
    for i, (s, c, f) in enumerate(items):
        keys = ["md5:" + hashlib.md5(f.read_bytes()).hexdigest()]
        g = read_gray(f)
        if g is None:
            unreadable.add(i)
        else:
            keys.append("ah:" + ahash(g))
            if is_random_noise(g):
                noise.add(i)
        for k in keys:
            if k in first_by_key:
                uf.union(first_by_key[k], i)
            else:
                first_by_key[k] = i
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(items)}")

    groups = defaultdict(list)
    for i in range(len(items)):
        groups[uf.find(i)].append(i)

    # ---- 3. Chọn ảnh giữ lại ----
    removed = []
    keep_by_class = defaultdict(list)
    for members in groups.values():
        cls_set = {items[i][1] for i in members}
        bad = [i for i in members if i in noise or i in unreadable]
        good = [i for i in members if i not in noise and i not in unreadable]
        for i in bad:
            removed.append((items[i], "noise" if i in noise else "unreadable"))
        if not good:
            continue
        if len(cls_set) > 1:
            for i in good:
                removed.append((items[i], "conflicting_labels:" + "/".join(sorted(cls_set))))
            continue
        # ưu tiên giữ bản trong train (thường là bản gốc), còn lại là bản trùng
        good.sort(key=lambda i: (SPLITS.index(items[i][0]), str(items[i][2])))
        keep_by_class[items[good[0]][1]].append(good[0])
        for i in good[1:]:
            removed.append((items[i], "duplicate"))

    # ---- 4. Chia lại stratified ----
    rng = np.random.default_rng(seed)
    rows = []
    for c in classes:
        idx = sorted(keep_by_class.get(c, []), key=lambda i: str(items[i][2]))
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
        n_test = max(1, int(round(n * ratios[2]))) if n >= 3 else 0
        n_train = n - n_val - n_test
        parts = {"train": idx[:n_train], "validation": idx[n_train:n_train + n_val],
                 "test": idx[n_train + n_val:]}
        for split, ids in parts.items():
            out_dir = RAW / split / c
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in ids:
                src_split, _, f = items[i]
                shutil.copy2(f, out_dir / f"{src_split}_{f.name}")
        rows.append({"class": c, "unique_images": n, "train": n_train,
                     "validation": n_val, "test": n - n_train - n_val})

    summary = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "reports" / "clean_split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"split": s, "class": c, "file": str(f.relative_to(ORIG)), "reason": r}
                  for (s, c, f), r in removed]).to_csv(
        ROOT / "reports" / "clean_split_removed.csv", index=False, encoding="utf-8-sig")

    reasons = defaultdict(int)
    for _, r in removed:
        reasons[r.split(":")[0]] += 1
    print("\n===== KẾT QUẢ LÀM SẠCH =====")
    print(f"  Ảnh gốc               : {len(items)}")
    print(f"  Bỏ do trùng lặp       : {reasons['duplicate']}")
    print(f"  Bỏ do nhãn mâu thuẫn  : {reasons['conflicting_labels']}")
    print(f"  Bỏ do ảnh nhiễu       : {reasons['noise']}")
    print(f"  Bỏ do ảnh hỏng        : {reasons['unreadable']}")
    print(f"  Ảnh duy nhất còn lại  : {int(summary.unique_images.sum())}")
    print(f"\n  Chia lại (seed {seed}, tỉ lệ {ratios}):")
    for s in SPLITS:
        print(f"    {s:<11}: {int(summary[s].sum())}")
    print(f"  Ít ảnh nhất: {summary.loc[summary.unique_images.idxmin(), 'class']} "
          f"({int(summary.unique_images.min())} ảnh)")
    print(f"\nDataset sạch: {RAW}\nDataset gốc : {ORIG}")
    print("Tiếp theo: python scripts\\check_duplicates.py  (kiểm tra lại, phải không còn trùng)")


if __name__ == "__main__":
    main()
