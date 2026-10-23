"""
src/training/train_svm.py
-------------------------
NGƯỜI LÀM: TN 3

Train + tune + đánh giá model Classical SVM (HOG + Color Histogram -> Scaler -> PCA -> SVM),
xuất kết quả CÙNG FORMAT với baseline_cnn / mobilenetv2_transfer để so sánh công bằng.

Dùng code của nhóm:
    - TN 2: src/features/hog_features.py  (batch_extract_features — 8196 chiều)
    - TN 2: src/features/pca_wrapper.py   (PCAWrapper, standardize=True)
    - TN 1: src/models/classical_svm.py   (ClassicalSVMClassifier — GridSearchCV)
    - C, gamma, số chiều PCA, số fold CV lấy từ configs/config.py -> CONFIG["svm"]
    --backend sklearn: bản dự phòng chỉ dùng hog_features.py + sklearn thuần.

Không import TensorFlow -> chạy được trên máy chỉ cài scikit-learn.

Cách chạy (từ thư mục gốc project):
    # 1) Test nhanh 5 class, 30 ảnh/class (~1-2 phút) để kiểm tra pipeline
    python -m src.training.train_svm --classes 5 --max-per-class 30 --tag quick

    # 2) Chỉ trích đặc trưng + đo thời gian (cache lại để lần sau không tính lại)
    python -m src.training.train_svm --features-only

    # 3) Train đầy đủ 36 class bằng code TN 1 + TN 2 (mặc định)
    python -m src.training.train_svm

Output (reports/ và checkpoints/):
    reports/classical_svm_evaluation.json      (cùng format *_evaluation.json của CNN)
    reports/classical_svm_summary.json
    reports/classical_svm_metadata.json
    reports/classical_svm_per_class_metrics.csv
    reports/classical_svm_gridsearch.csv
    reports/figures/classical_svm_confusion_matrix.png
    reports/figures/classical_svm_gridsearch.png
    reports/figures/classical_svm_misclassified.png
    checkpoints/classical_svm.pkl              (bundle cho Streamlit, xem predict_image())
    checkpoints/pca_model.pkl + svm_model.pkl  (cùng format run_full_pipeline của TN 1)
    checkpoints/features/*.npz                 (cache đặc trưng)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from configs.config import CONFIG

MODEL_NAME = "classical_svm"
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # giống src/data/dataset.py


# =====================================================================
# 1. Đọc danh sách ảnh — CÙNG logic với src/data/dataset.py
#    (thư mục class sort theo tên, file sort theo tên) => cùng tập test 409 ảnh
# =====================================================================
def list_split(split_dir: str, class_names: list[str] | None = None):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Không thấy thư mục {split_dir}. Kiểm tra data/raw/.")
    found = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if class_names is None:
        class_names = found
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cdir = split_dir / cls
        if not cdir.exists():
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMG_EXTENSIONS:
                paths.append(str(f))
                labels.append(idx)
    return paths, np.array(labels, dtype=int), class_names


def subsample(paths, labels, max_per_class: int | None, seed: int):
    if not max_per_class:
        return paths, labels
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.extend(sorted(idx.tolist()))
    keep = sorted(keep)
    return [paths[i] for i in keep], labels[keep]


# =====================================================================
# 2. Trích đặc trưng có cache (HOG chỉ tính 1 lần, GridSearch dùng lại)
# =====================================================================
def _feature_params() -> dict:
    from src.features import hog_features as hf
    return {
        "img_size": list(hf.DEFAULT_IMG_SIZE),
        "hog_orientations": hf.HOG_ORIENTATIONS,
        "hog_pixels_per_cell": list(hf.HOG_PIXELS_PER_CELL),
        "hog_cells_per_block": list(hf.HOG_CELLS_PER_BLOCK),
        "color_hist_bins": hf.COLOR_HIST_BINS,
        "color_weight": getattr(hf, "COLOR_WEIGHT", 1.0),
    }


def extract_features_cached(paths: list[str], split: str, cache_dir: Path, n_jobs: int = -1):
    """Trả về (X, keep_mask, seconds). Cache theo danh sách file + tham số HOG,
    nên đổi tham số trong hog_features.py (vd COLOR_WEIGHT) sẽ tự trích lại."""
    from src.features.hog_features import batch_extract_features

    params = _feature_params()
    key = hashlib.md5((json.dumps(params) + "|".join(paths)).encode()).hexdigest()[:10]
    cache_file = cache_dir / f"{split}_{key}.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        print(f"  Dùng cache {cache_file.name}: X={d['X'].shape}")
        return d["X"], d["keep"], float(d["seconds"])

    print(f"  Trích đặc trưng {len(paths)} ảnh ({split})...")
    t0 = time.time()
    try:
        X, keep = batch_extract_features(paths, show_progress=True, on_error="skip",
                                         return_valid_mask=True, n_jobs=n_jobs)
    except NotImplementedError:
        raise SystemExit("\n[DỪNG] src/features/hog_features.py chưa được implement (TN 2).")
    seconds = time.time() - t0
    for p, k in zip(paths, keep):
        if not k:
            print(f"  [!] Bỏ qua ảnh lỗi: {p}")
    X = X.astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, keep=keep, seconds=seconds)
    print(f"  Xong {split}: X={X.shape}, {seconds:.1f}s ({seconds / max(len(paths), 1) * 1000:.1f} ms/ảnh)")
    return X, keep, seconds


# =====================================================================
# 3. Train — 2 backend
# =====================================================================
def _cv_folds(y, wanted: int) -> int:
    min_count = int(np.bincount(y).min())
    folds = max(2, min(wanted, min_count))
    if folds < wanted:
        print(f"  [!] Class nhỏ nhất chỉ có {min_count} ảnh -> giảm CV xuống {folds} fold")
    return folds


def hellinger_color(Xc, w: float = 10.0, color_weight: float = 10.0):
    """Biến đổi Hellinger cho histogram màu: sqrt(tỉ lệ pixel) rồi nhân trọng số w.
    - Chia color_weight để lấy lại tỉ lệ pixel gốc (hog_features đã nhân COLOR_WEIGHT).
    - sqrt làm các bin nhỏ (màu ít xuất hiện) không bị các bin lớn (màu nền) lấn át
      -> khoảng cách Euclid trong kernel RBF phản ánh khác biệt màu tốt hơn.
    Đặt ở mức module để pickle được (Streamlit load lại model)."""
    return w * np.sqrt(np.clip(Xc / color_weight, 0, None))


def train_sklearn(X, y, args):
    """Cấu hình đã tune (mặc định):
        HOG (8100 chiều) --PCA--> 50/100 chiều  ┐
        Màu (96 chiều)   --Hellinger x w-------┴--> SVC RBF
    Không StandardScaler (làm mất thông tin màu), không CalibratedClassifierCV khi
    GridSearch (làm dự đoán sai lệch). GridSearch với probability=False (nhanh), sau đó
    refit model tốt nhất với probability=True để Streamlit có confidence/Top-K."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.svm import SVC

    # Import lại qua tên module đầy đủ để file .pkl lưu "src.training.train_svm.hellinger_color"
    # (nếu chạy bằng python -m, hàm trong file này mang tên __main__ -> Streamlit không load được)
    from src.training.train_svm import hellinger_color as _hellinger

    fp = _feature_params()
    n_color = 3 * fp["color_hist_bins"]
    n_hog = X.shape[1] - n_color
    features = ColumnTransformer([
        ("hog", PCA(random_state=args.seed), list(range(n_hog))),
        ("color", FunctionTransformer(_hellinger,
                                      kw_args={"color_weight": fp["color_weight"]}),
         list(range(n_hog, X.shape[1]))),
    ])
    pipe = Pipeline([("features", features),
                     ("svc", SVC(kernel="rbf", random_state=args.seed))])
    hog_pcas = [min(n, X.shape[0] - 1) for n in args.hog_pca]
    grid = {
        "features__hog__n_components": hog_pcas,
        "features__color__kw_args": [{"w": w, "color_weight": fp["color_weight"]} for w in args.color_w],
        "svc__C": args.c,
        "svc__gamma": args.gamma,
    }
    folds = _cv_folds(y, args.cv)
    n_comb = len(hog_pcas) * len(args.color_w) * len(args.c) * len(args.gamma)
    print(f"  GridSearchCV: HOG-PCA {hog_pcas} x color_w {args.color_w} x C {args.c} x gamma {args.gamma}")
    print(f"  = {n_comb} tổ hợp x {folds} fold = {n_comb * folds} lần fit")

    gs = GridSearchCV(
        pipe, grid, scoring="f1_macro", n_jobs=args.n_jobs, verbose=1,
        cv=StratifiedKFold(folds, shuffle=True, random_state=args.seed),
        return_train_score=True,
    )
    t0 = time.time()
    gs.fit(X, y)
    t_search = time.time() - t0

    best = gs.best_estimator_
    best.set_params(svc__probability=True)
    t1 = time.time()
    import warnings
    with warnings.catch_warnings():  # sklearn >= 1.9 báo deprecated probability=True, vẫn chạy bình thường
        warnings.simplefilter("ignore", FutureWarning)
        best.fit(X, y)
    t_refit = time.time() - t1

    hog_pca = best.named_steps["features"].named_transformers_["hog"]
    bp = gs.best_params_
    best_params = {"C": bp["svc__C"], "gamma": bp["svc__gamma"],
                   "hog_pca": bp["features__hog__n_components"],
                   "color_w": bp["features__color__kw_args"]["w"]}
    bundle = {
        "backend": "sklearn",
        "pipeline": best,
        "explained_variance": float(hog_pca.explained_variance_ratio_.sum()),
        "n_support_vectors": _n_support(best.named_steps["svc"]),
        "n_model_dims": int(best_params["hog_pca"] + n_color),
    }
    cr = gs.cv_results_
    cv_results = pd.DataFrame({
        "hog_pca": cr["param_features__hog__n_components"],
        "color_w": [d["w"] for d in cr["param_features__color__kw_args"]],
        "C": cr["param_svc__C"], "gamma": cr["param_svc__gamma"],
        "mean_test_score": cr["mean_test_score"], "std_test_score": cr["std_test_score"],
        "mean_train_score": cr["mean_train_score"], "mean_fit_time": cr["mean_fit_time"],
        "rank_test_score": cr["rank_test_score"],
    }).sort_values("rank_test_score")
    return bundle, best_params, float(gs.best_score_), cv_results, t_search, t_refit


def _n_support(model) -> int:
    """Số support vector, kể cả khi SVC bị bọc trong CalibratedClassifierCV (code TN 1)."""
    if hasattr(model, "support_vectors_"):
        return int(model.support_vectors_.shape[0])
    for cc in getattr(model, "calibrated_classifiers_", []):
        inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if hasattr(inner, "support_vectors_"):
            return int(inner.support_vectors_.shape[0])
    return 0


def train_tn1(X, y, args):
    """Dùng code của nhóm: PCAWrapper (TN 2, standardize=True) + ClassicalSVMClassifier (TN 1).
    Chuẩn hoá (StandardScaler) trước PCA vì SVM RBF nhạy với thang đo."""
    from src.features.pca_wrapper import PCAWrapper
    from src.models.classical_svm import ClassicalSVMClassifier

    n_comp = min(args.n_components, X.shape[0] - 1, X.shape[1])
    pca = PCAWrapper(n_components=n_comp, standardize=True)
    Xp = pca.fit_transform(X)

    clf = ClassicalSVMClassifier(c_values=args.c, gamma_values=args.gamma, cv_folds=_cv_folds(y, args.cv))
    t0 = time.time()
    info = clf.train(Xp, y)
    t_search = time.time() - t0

    bundle = {
        "backend": "tn1",
        "pca": pca,
        "svm": clf,
        "explained_variance": float(pca.get_total_variance_kept()),
        "n_support_vectors": _n_support(clf.model),
    }
    return bundle, info["best_params"], float(info["best_cv_score"]), None, t_search, 0.0


# =====================================================================
# 4. Dự đoán từ bundle (dùng chung cho đánh giá + Streamlit của TN 4)
# =====================================================================
def bundle_predict(bundle, X):
    if bundle["backend"] == "sklearn":
        return bundle["pipeline"].predict(X)
    return bundle["svm"].predict(bundle["pca"].transform(X))


def bundle_predict_proba(bundle, X):
    if bundle["backend"] == "sklearn":
        return bundle["pipeline"].predict_proba(X)
    return bundle["svm"].predict_proba(bundle["pca"].transform(X))


def load_svm_bundle(path: str | None = None):
    path = path or Path(CONFIG["paths"]["checkpoints_dir"]) / f"{MODEL_NAME}.pkl"
    return joblib.load(path)


def predict_image(bundle, image_path: str, top_k: int = 3) -> list[dict]:
    """Cho TN 4 (Streamlit):
        bundle = load_svm_bundle()
        predict_image(bundle, "anh.jpg")  ->  [{"class": "carrot", "confidence": 0.91}, ...]
    """
    from src.features.hog_features import extract_combined_features
    x = extract_combined_features(image_path).reshape(1, -1)
    proba = bundle_predict_proba(bundle, x)[0]
    top = np.argsort(proba)[::-1][:top_k]
    return [{"class": bundle["class_names"][i], "confidence": float(proba[i])} for i in top]


# =====================================================================
# 5. Đánh giá — cùng key với src/evaluation/metrics.py
# =====================================================================
def compute_metrics(y_true, y_pred, class_names):
    from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                                 precision_score, recall_score)
    labels = list(range(len(class_names)))
    kw = dict(labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", **kw)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", **kw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **kw)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
    }


def detect_overfitting(train_acc, val_acc, threshold=0.1):
    # Giống src/evaluation/metrics.py để so sánh cùng tiêu chí
    gap = train_acc - val_acc
    if gap > threshold:
        status = "overfitting"
        exp = (f"Train accuracy ({train_acc:.2%}) cao hơn val accuracy ({val_acc:.2%}) khoảng {gap:.2%}, "
               f"vượt ngưỡng {threshold:.0%} -> mô hình có dấu hiệu overfitting.")
    elif train_acc < 0.5 and val_acc < 0.5:
        status = "underfitting"
        exp = f"Cả train ({train_acc:.2%}) và val ({val_acc:.2%}) đều thấp -> underfitting."
    else:
        status = "good_fit"
        exp = (f"Gap giữa train ({train_acc:.2%}) và val accuracy ({val_acc:.2%}) chỉ {gap:.2%}, "
               "trong ngưỡng chấp nhận được -> mô hình tổng quát hoá tốt.")
    return {"status": status, "gap": float(gap), "explanation": exp}


def plot_gridsearch(cv_results: pd.DataFrame, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    col = "color_w" if "color_w" in cv_results else "gamma"
    table = cv_results.assign(**{col: cv_results[col].astype(str)}).pivot_table(
        index="C", columns=col, values="mean_test_score", aggfunc="max")
    fig, ax = plt.subplots(figsize=(1.6 * len(table.columns) + 3, 0.8 * len(table) + 2.5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="Greens", ax=ax, cbar_kws={"label": "F1 macro (CV)"})
    ax.set_title(f"GridSearchCV — F1 macro (CV) theo C và {col} (tốt nhất của các tham số còn lại)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_misclassified(paths, y_true, y_pred, class_names, save_path: Path, n=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.features.hog_features import _read_image_rgb
    wrong = np.where(y_true != y_pred)[0][:n]
    if len(wrong) == 0:
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        try:
            ax.imshow(_read_image_rgb(paths[i]))
        except Exception:
            pass
        ax.set_title(f"Thật: {class_names[y_true[i]]}\nĐoán: {class_names[y_pred[i]]}", fontsize=9, color="red")
    fig.suptitle("Classical SVM — ví dụ dự đoán sai", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =====================================================================
# 6. Main
# =====================================================================
def parse_gamma(values):
    out = []
    for v in values:
        try:
            out.append(float(v))
        except ValueError:
            out.append(v)  # 'scale' / 'auto'
    return out


def main():
    p = argparse.ArgumentParser(description="Train + đánh giá Classical SVM (TN 3)")
    svm_cfg = CONFIG.get("svm", {})
    p.add_argument("--backend", choices=["sklearn", "tn1"], default="sklearn",
                   help="sklearn (mặc định, đã tune): HOG-PCA + màu Hellinger -> SVC. "
                        "tn1: PCAWrapper + ClassicalSVMClassifier gốc của nhóm (để so sánh)")
    p.add_argument("--hog-pca", type=int, nargs="+", default=[50, 100],
                   help="Số chiều PCA cho phần HOG (backend sklearn)")
    p.add_argument("--color-w", type=float, nargs="+", default=[10, 20],
                   help="Trọng số phần màu sau biến đổi Hellinger (backend sklearn)")
    p.add_argument("--c", type=float, nargs="+", default=[3, 10, 30])
    p.add_argument("--gamma", nargs="+", default=["scale"])
    p.add_argument("--n-components", type=int, default=svm_cfg.get("pca_n_components", 200))
    p.add_argument("--cv", type=int, default=svm_cfg.get("cv_folds", 5))
    p.add_argument("--max-per-class", type=int, default=None, help="Subsample tập train (để tune nhanh)")
    p.add_argument("--classes", type=int, default=None, help="Chỉ dùng N class đầu (test nhanh)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features-only", action="store_true", help="Chỉ trích + cache đặc trưng, đo thời gian")
    p.add_argument("--tag", type=str, default="", help="Hậu tố tên output, vd 'quick' -> classical_svm_quick_*")
    args = p.parse_args()
    args.gamma = parse_gamma(args.gamma)

    name = MODEL_NAME + (f"_{args.tag}" if args.tag else "")
    paths_cfg = CONFIG["paths"]
    reports_dir, figures_dir = Path(paths_cfg["reports_dir"]), Path(paths_cfg["figures_dir"])
    ckpt_dir = Path(paths_cfg["checkpoints_dir"])
    cache_dir = ckpt_dir / "features"
    for d in (reports_dir, figures_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load danh sách ảnh ----
    tr_paths, y_tr, class_names = list_split(paths_cfg["train_dir"])
    if args.classes:
        class_names = class_names[: args.classes]
        tr_paths, y_tr, _ = list_split(paths_cfg["train_dir"], class_names)
    va_paths, y_va, _ = list_split(paths_cfg["val_dir"], class_names)
    te_paths, y_te, _ = list_split(paths_cfg["test_dir"], class_names)
    tr_paths, y_tr = subsample(tr_paths, y_tr, args.max_per_class, args.seed)
    print(f"Classes: {len(class_names)} | train {len(tr_paths)} | val {len(va_paths)} | test {len(te_paths)}")

    # ---- Đặc trưng ----
    X_tr, k_tr, s_tr = extract_features_cached(tr_paths, "train", cache_dir, args.n_jobs)
    X_va, k_va, _ = extract_features_cached(va_paths, "validation", cache_dir, args.n_jobs)
    X_te, k_te, s_te = extract_features_cached(te_paths, "test", cache_dir, args.n_jobs)
    y_tr, y_va, y_te = y_tr[k_tr], y_va[k_va], y_te[k_te]
    te_paths = [p for p, k in zip(te_paths, k_te) if k]
    print(f"Số chiều đặc trưng: {X_tr.shape[1]}")
    if (~k_te).any():
        print(f"[!] CẢNH BÁO: {(~k_te).sum()} ảnh test bị bỏ -> số ảnh test khác CNN, ghi rõ trong báo cáo.")

    if args.features_only:
        per_img = s_tr / max(len(X_tr), 1)
        print(f"\nThời gian trích đặc trưng: {per_img * 1000:.1f} ms/ảnh (train {s_tr:.0f}s, test {s_te:.0f}s)")
        return

    # ---- Train ----
    trainer = train_sklearn if args.backend == "sklearn" else train_tn1
    bundle, best_params, cv_score, cv_results, t_search, t_refit = trainer(X_tr, y_tr, args)
    bundle.update({"class_names": class_names, "feature_params": _feature_params(),
                   "n_features": int(X_tr.shape[1]), "n_components": args.n_components})
    print(f"Best params: {best_params} | CV F1 macro: {cv_score:.4f} | search {t_search / 60:.1f} phút")

    # ---- Đánh giá ----
    train_acc = float((bundle_predict(bundle, X_tr) == y_tr).mean())
    val_acc = float((bundle_predict(bundle, X_va) == y_va).mean())
    t0 = time.time()
    y_pred = bundle_predict(bundle, X_te)
    clf_ms = (time.time() - t0) / len(X_te) * 1000
    feat_ms = s_te / max(len(X_te), 1) * 1000 if s_te else None

    metrics = compute_metrics(y_te, y_pred, class_names)
    with open(reports_dir / f"{name}_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    rep = metrics["classification_report"]
    per_class = pd.DataFrame([
        {"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
         "f1_score": rep[c]["f1-score"], "support": rep[c]["support"]} for c in class_names
    ]).sort_values("f1_score").reset_index(drop=True)
    per_class.to_csv(reports_dir / f"{name}_per_class_metrics.csv", index=False)

    from src.evaluation.confusion_matrix import plot_confusion_matrix, top_confused_pairs
    import matplotlib
    matplotlib.use("Agg")
    fig, cm = plot_confusion_matrix(y_te, y_pred, class_names,
                                    save_path=str(figures_dir / f"{name}_confusion_matrix.png"))
    matplotlib.pyplot.close(fig)
    plot_misclassified(te_paths, y_te, y_pred, class_names, figures_dir / f"{name}_misclassified.png")
    if cv_results is not None:
        cv_results.to_csv(reports_dir / f"{name}_gridsearch.csv", index=False)
        plot_gridsearch(cv_results, figures_dir / f"{name}_gridsearch.png")

    overfit = detect_overfitting(train_acc, val_acc)
    summary = {
        "model_name": name,
        "test_accuracy": metrics["accuracy"],
        "test_precision_macro": metrics["precision_macro"],
        "test_recall_macro": metrics["recall_macro"],
        "test_f1_macro": metrics["f1_macro"],
        "test_f1_weighted": metrics["f1_weighted"],
        "num_test_samples": int(len(y_te)),
        "best_class": per_class.iloc[-1]["class"],
        "worst_class": per_class.iloc[0]["class"],
        "top_confused_pairs": top_confused_pairs(cm, class_names)[:5],
        "num_misclassified": int((y_te != y_pred).sum()),
        "overfitting_analysis": overfit,
    }
    with open(reports_dir / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fp = bundle["feature_params"]
    metadata = {
        "model_name": name,
        "architecture": (
            f"HOG({fp['hog_orientations']} hướng, cell {fp['hog_pixels_per_cell']}) -> PCA({best_params.get('hog_pca')}) "
            f"+ Color Hist RGB({fp['color_hist_bins']} bins, Hellinger x{best_params.get('color_w')}) -> SVC RBF"
            if args.backend == "sklearn" else
            f"HOG + Color Hist -> StandardScaler -> PCA({args.n_components}) -> "
            f"CalibratedClassifierCV(SVC RBF) [code TN 1]"),
        "backend": args.backend,
        # report.py cần số nguyên: dùng số tham số lưu trong support vectors
        "num_parameters": int(bundle["n_support_vectors"] * bundle.get("n_model_dims", min(args.n_components, X_tr.shape[1]))),
        "num_parameters_note": "SVM không có trọng số như CNN; giá trị = số support vector x số chiều đầu vào SVM",
        "n_support_vectors": bundle["n_support_vectors"],
        "input_size": fp["img_size"][0],
        "n_features_before_pca": int(X_tr.shape[1]),
        "pca_components": best_params.get("hog_pca", min(args.n_components, X_tr.shape[1])),
        "pca_explained_variance": bundle["explained_variance"],
        "optimizer": "GridSearchCV (scoring = f1_macro)",
        "loss_function": "hinge (SVM)",
        "param_grid": ({"C": args.c, "gamma": args.gamma, "hog_pca": args.hog_pca, "color_w": args.color_w}
                       if args.backend == "sklearn" else {"C": args.c, "gamma": args.gamma}),
        "cv_folds": args.cv,
        "best_params": best_params,
        "best_cv_f1_macro": cv_score,
        "num_train_samples": int(len(y_tr)),
        "max_per_class": args.max_per_class,
        "training_time_seconds": round(t_search + t_refit, 1),
        "gridsearch_time_seconds": round(t_search, 1),
        "feature_extraction_time_seconds": round(s_tr, 1),
        "inference_ms_per_image": round((feat_ms or 0) + clf_ms, 2),
        "inference_ms_per_image_classifier_only": round(clf_ms, 3),
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
        "class_weight_applied": False,
        "class_names": class_names,
    }
    with open(reports_dir / f"{name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    joblib.dump(bundle, ckpt_dir / f"{name}.pkl")
    if args.backend == "tn1" and not args.tag:
        # Lưu thêm đúng format run_full_pipeline của TN 1 để Streamlit (TN 4) dùng được
        bundle["pca"].save(str(ckpt_dir / "pca_model.pkl"))
        bundle["svm"].classes = class_names
        bundle["svm"].save(str(ckpt_dir / "svm_model.pkl"))

    print("\n" + "=" * 60)
    print(f"[{name}] Test accuracy {metrics['accuracy']:.2%} | F1 macro {metrics['f1_macro']:.4f}")
    print(f"Train acc {train_acc:.2%} | Val acc {val_acc:.2%} -> {overfit['status']}")
    print(f"Best: {best_params} | PCA (HOG) giữ {bundle['explained_variance']:.1%} phương sai")
    print(f"Đã lưu: reports/{name}_*.json, checkpoints/{name}.pkl")
    print("Tiếp theo: python -m src.evaluation.compare_all")


if __name__ == "__main__":
    main()
