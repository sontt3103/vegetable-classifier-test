"""
scripts/svm_experiments.py
--------------------------
Thí nghiệm cải thiện đặc trưng cho SVM (HOG + PCA + SVM) – Hướng A.

Vấn đề hiện tại (train_svm.py): train 98% nhưng validation chỉ ~32% -> overfitting.
GridSearch trên C/gamma không giúp được (mọi tổ hợp F1 CV 0.28-0.33), nên nút thắt
là ĐẶC TRƯNG. Script này thử lần lượt các cải tiến đặc trưng, mỗi bước thêm 1 thay đổi
(thí nghiệm tích lũy / ablation) để thấy rõ đóng góp của từng thay đổi:

    E0  HOG(cell 8) -> PCA50 + histogram RGB          (tương đương cấu hình TN 3)
    E1  thay RGB bằng histogram HSV                    (Hue ít phụ thuộc ánh sáng)
    E2  + histogram HSV theo vùng (lưới 2x2 + vùng giữa) (giảm ảnh hưởng của nền)
    E3  + đặc trưng kết cấu LBP                        (vỏ sần, vân, hạt)
    E4  thay HOG cell 8 bằng HOG thô cell 16           (ít chiều nhiễu hơn)
    E5  + lật ngang ảnh train (nhân đôi dữ liệu)
    E6  + class_weight='balanced', mở rộng lưới C
    --- Hướng B (cần TensorFlow; bỏ qua bằng --no-cnn) ---
    E7  Đặc trưng MobileNetV2 đã huấn luyện sẵn (KHÔNG fine-tune) -> PCA -> SVM
    E8  Đặc trưng MobileNetV2 + toàn bộ đặc trưng thủ công của E5

Quy trình đúng (không rò rỉ):
    - Siêu tham số (C, class_weight) chọn bằng GridSearchCV 5-fold CHỈ trên tập train.
    - Cấu hình tốt nhất được chọn theo F1 macro trên tập VALIDATION.
    - Tập TEST chỉ được báo cáo cho DUY NHẤT cấu hình được chọn.

Các khối đặc trưng và pipeline nằm ở src/features/svm_features.py (dùng chung với
train_svm.py và svm_predict.py). Cấu hình E5 đã được chọn làm mặc định cho train_svm.py.

Cách biểu diễn: mọi histogram dùng biến đổi Hellinger (căn bậc hai tỉ lệ pixel); mỗi khối
đặc trưng được chuẩn hóa L2 theo từng ảnh rồi mới ghép, để không khối nào lấn át khối nào
(thay cho StandardScaler – vốn làm mất thông tin màu, xem reports/svm_tuning_log.csv).

Chạy (sau khi đã có dataset sạch ở data/raw):
    python scripts/svm_experiments.py            # E0-E8
    python scripts/svm_experiments.py --no-cnn   # chỉ E0-E6 (không cần TensorFlow)
Kết quả:
    reports/svm_feature_experiments.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG  # noqa: E402
from src.features import svm_features as sf  # noqa: E402
from src.features.hog_features import load_and_resize  # noqa: E402
from src.features.svm_features import extract_blocks  # noqa: E402
from src.training.train_svm import list_split  # noqa: E402

IMG_SIZE = tuple(CONFIG["svm"]["img_size"])   # (128, 128) – giống pipeline SVM hiện tại
SEED = CONFIG["data"]["seed"]
CV_FOLDS = CONFIG["svm"]["cv_folds"]
HOG_PCA = 50


# =====================================================================
# 1. Đặc trưng: dùng chung src/features/svm_features.py (không viết lại)
# =====================================================================
def _load_one(path):
    try:
        rgb = load_and_resize(path, IMG_SIZE)
    except Exception:  # noqa: BLE001
        return None
    return extract_blocks(rgb), extract_blocks(np.ascontiguousarray(rgb[:, ::-1]))


def extract_split(paths):
    res = Parallel(n_jobs=-1)(delayed(_load_one)(p) for p in paths)
    keep = np.array([r is not None for r in res])
    res = [r for r in res if r is not None]
    names = res[0][0].keys()
    orig = {k: np.stack([r[0][k] for r in res]) for k in names}
    flip = {k: np.stack([r[1][k] for r in res]) for k in names}
    return orig, flip, keep


def build_pipeline(blocks, dims):
    return sf.build_pipeline(blocks, dims, hog_pca=HOG_PCA, seed=SEED)


def stack(feats: dict, blocks: list[str]) -> np.ndarray:
    return np.hstack([feats[b] for b in blocks])


EXPERIMENTS = [
    # (mã, mô tả, khối đặc trưng, lật ảnh, lưới tham số)
    ("E0", "HOG8-PCA50 + RGB (tương đương TN3)", ["hog8", "rgb"], False, {"svc__C": [1, 3, 10, 30]}),
    ("E1", "HOG8-PCA50 + HSV", ["hog8", "hsv"], False, {"svc__C": [1, 3, 10, 30]}),
    ("E2", "+ HSV theo vùng", ["hog8", "hsv", "hsv_grid"], False, {"svc__C": [1, 3, 10, 30]}),
    ("E3", "+ LBP", ["hog8", "hsv", "hsv_grid", "lbp"], False, {"svc__C": [1, 3, 10, 30]}),
    ("E4", "HOG thô (cell 16) thay HOG8", ["hog16", "hsv", "hsv_grid", "lbp"], False, {"svc__C": [1, 3, 10, 30]}),
    ("E5", "+ lật ngang ảnh train", ["hog16", "hsv", "hsv_grid", "lbp"], True, {"svc__C": [1, 3, 10, 30]}),
    ("E6", "+ class_weight, lưới C rộng", ["hog16", "hsv", "hsv_grid", "lbp"], True,
     {"svc__C": [0.3, 1, 3, 10, 30], "svc__class_weight": [None, "balanced"]}),
    ("E7", "Đặc trưng MobileNetV2 -> PCA (hướng B)", ["cnn"], True,
     {"svc__C": [1, 3, 10, 30], "features__cnn__pca__n_components": [128, 256]}),
    ("E8", "MobileNetV2 + đặc trưng thủ công E5", ["cnn", "hog16", "hsv", "hsv_grid", "lbp"], True,
     {"svc__C": [1, 3, 10, 30], "features__cnn__pca__n_components": [128, 256]}),
]


# =====================================================================
# 3. Chạy thí nghiệm
# =====================================================================
def main():
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    paths_cfg = CONFIG["paths"]
    tr_p, y_tr, class_names = list_split(paths_cfg["train_dir"])
    va_p, y_va, _ = list_split(paths_cfg["val_dir"], class_names)
    te_p, y_te, _ = list_split(paths_cfg["test_dir"], class_names)
    print(f"Classes: {len(class_names)} | train {len(tr_p)} | val {len(va_p)} | test {len(te_p)}")

    t0 = time.time()
    tr, tr_flip, k_tr = extract_split(tr_p)
    y_tr = y_tr[k_tr]
    va, _, k_va = extract_split(va_p)
    y_va = y_va[k_va]
    te, _, k_te = extract_split(te_p)
    y_te = y_te[k_te]
    experiments = EXPERIMENTS
    if "--no-cnn" in sys.argv:
        experiments = [e for e in EXPERIMENTS if "cnn" not in e[2]]
    else:
        try:
            print("Trích đặc trưng MobileNetV2 (ảnh 224x224, kèm ảnh lật cho train)...")
            for feats, flip_feats, paths, keep in ((tr, tr_flip, tr_p, k_tr), (va, None, va_p, k_va),
                                                   (te, None, te_p, k_te)):
                Xc, Xcf, kc = sf._cnn_for_paths([p for p, k in zip(paths, keep) if k], flip_feats is not None, -1)
                if not kc.all():
                    raise RuntimeError("Có ảnh đọc được cho đặc trưng thủ công nhưng lỗi khi đọc cho CNN")
                feats["cnn"] = Xc
                if flip_feats is not None:
                    flip_feats["cnn"] = Xcf
        except Exception as e:  # noqa: BLE001 – thiếu TensorFlow / không tải được trọng số ImageNet
            print(f"[!] Bỏ qua E7, E8 (hướng B): {e}")
            experiments = [e for e in EXPERIMENTS if "cnn" not in e[2]]
    dims = {b: v.shape[1] for b, v in tr.items()}
    print(f"Trích đặc trưng xong trong {time.time() - t0:.0f}s | số chiều mỗi khối: {dims}\n")

    rows, models = [], {}
    for code, desc, blocks, use_flip, grid in experiments:
        X = stack(tr, blocks)
        y, groups = y_tr, np.arange(len(y_tr))
        if use_flip:   # ảnh lật nằm cùng fold với ảnh gốc -> không rò rỉ trong CV
            X = np.vstack([X, stack(tr_flip, blocks)])
            y, groups = np.concatenate([y_tr, y_tr]), np.concatenate([groups, groups])

        gs = GridSearchCV(build_pipeline(blocks, dims), grid, scoring="f1_macro", n_jobs=-1,
                          cv=StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                          return_train_score=True)
        t1 = time.time()
        gs.fit(X, y, groups=groups)
        model = gs.best_estimator_
        Xtr, Xva = stack(tr, blocks), stack(va, blocks)
        p_tr, p_va = model.predict(Xtr), model.predict(Xva)
        row = {
            "code": code, "mo_ta": desc,
            "so_chieu": sf.model_dims(blocks, dims, HOG_PCA,
                                      gs.best_params_.get("features__cnn__pca__n_components", 0)),
            "best_params": {k.replace("svc__", "").replace("features__cnn__pca__n_components", "cnn_pca"): v
                            for k, v in gs.best_params_.items()},
            "cv_f1": round(gs.best_score_, 4),
            "train_acc": round(accuracy_score(y_tr, p_tr), 4),
            "val_acc": round(accuracy_score(y_va, p_va), 4),
            "val_f1": round(f1_score(y_va, p_va, average="macro"), 4),
            "giay": round(time.time() - t1, 1),
        }
        rows.append(row)
        models[code] = (model, blocks)
        print(f"{code:3s} {desc:38s} | CV F1 {row['cv_f1']:.3f} | train {row['train_acc']:.3f} | "
              f"val acc {row['val_acc']:.3f} | val F1 {row['val_f1']:.3f} | {row['best_params']}")

    df = pd.DataFrame(rows)
    best = df.sort_values(["val_f1", "cv_f1"], ascending=False).iloc[0]
    model, blocks = models[best["code"]]
    p_te = model.predict(stack(te, blocks))
    df["test_acc"] = np.nan   # để trống cho các cấu hình không được chọn
    df["test_f1"] = np.nan
    df.loc[df["code"] == best["code"], "test_acc"] = round(accuracy_score(y_te, p_te), 4)
    df.loc[df["code"] == best["code"], "test_f1"] = round(f1_score(y_te, p_te, average="macro"), 4)

    out = ROOT / "reports" / "svm_feature_experiments.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print(f"Cấu hình tốt nhất theo F1 VALIDATION: {best['code']} – {best['mo_ta']}")
    print(f"  TEST (chỉ báo cho cấu hình này): accuracy {accuracy_score(y_te, p_te):.4f} | "
          f"F1 macro {f1_score(y_te, p_te, average='macro'):.4f}")
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
