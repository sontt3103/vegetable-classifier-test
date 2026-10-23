"""
src/models/classical_svm.py
----------------------------
NGƯỜI LÀM: TN 1

Mục đích:
    Model chính của môn học — dùng SVM để phân loại ảnh rau củ quả.

    Pipeline hoàn chỉnh:
        Ảnh → HOG + Color Histogram (TN 2) → PCA giảm chiều (TN 2)
            → SVM phân loại (file này)

    SVM là gì?
        SVM (Support Vector Machine) tìm đường biên (hyperplane) để chia
        dữ liệu thành các nhóm. Với kernel RBF, SVM có thể tạo ra đường biên
        phi tuyến tính, phù hợp với dữ liệu ảnh phức tạp.

    GridSearchCV là gì?
        Tự động thử nhiều tổ hợp C và gamma để tìm ra tổ hợp tốt nhất.
        Ví dụ thử: C=[0.1, 1, 10, 100] × gamma=['scale', 'auto'] = 8 tổ hợp.
        Mỗi tổ hợp dùng cross-validation để đánh giá → tốn thời gian nhưng đáng.

Phụ thuộc (TN 2 đã làm xong):
    from src.features.hog_features import batch_extract_features
    from src.features.pca_wrapper import PCAWrapper

Cách dùng:
    # Train từ đầu
    python -m src.models.classical_svm --data-dir data/raw

    # Hoặc dùng run_full_pipeline() trong code:
    from src.models.classical_svm import run_full_pipeline
    run_full_pipeline(data_dir="data/raw")
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np


class ClassicalSVMClassifier:
    """
    SVM classifier với pipeline đầy đủ: fit → predict → save/load.
    """

    def __init__(
        self,
        c_values: list = None,
        gamma_values: list = None,
        cv_folds: int = None,
    ):
        """
        Args:
            c_values: Danh sách giá trị C để thử (C điều chỉnh độ rộng margin)
                      C nhỏ → margin rộng, có thể sai nhiều hơn
                      C lớn → margin hẹp, ôm sát dữ liệu hơn nhưng dễ overfit
            gamma_values: Danh sách giá trị gamma để thử
                          'scale' và 'auto' là 2 cách tính tự động của sklearn
            cv_folds: Số fold cross-validation (thường dùng 5)
        """
        # Lấy giá trị từ config nếu không truyền vào
        try:
            from configs.config import CONFIG
            svm_cfg = CONFIG["svm"]
            if c_values is None:
                c_values = svm_cfg["c_list"]
            if gamma_values is None:
                gamma_values = svm_cfg["gamma_list"]
            if cv_folds is None:
                cv_folds = svm_cfg["cv_folds"]
        except (ImportError, KeyError):
            if c_values is None:
                c_values = [0.1, 1, 10, 100]
            if gamma_values is None:
                gamma_values = ["scale", "auto"]
            if cv_folds is None:
                cv_folds = 5

        self.c_values = c_values
        self.gamma_values = gamma_values
        self.cv_folds = cv_folds

        self.model = None         # SVC object sau khi train
        self.best_params = None   # C và gamma tốt nhất tìm được
        self.classes = None       # Tên các class (36 loại rau củ)
        self.is_fitted = False

    def _check_fitted(self, action: str = "dùng") -> None:
        if not self.is_fitted:
            raise RuntimeError(f"Cần gọi train() trước khi {action}. Model chưa được train.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> dict:
        """
        Train SVM với GridSearchCV để tìm hyperparameter tốt nhất.

        Args:
            X_train: Ma trận đặc trưng đã qua PCA, shape (n_samples, n_components)
                     Ví dụ: (5000, 200) — 5000 ảnh, mỗi ảnh 200 đặc trưng sau PCA
            y_train: Nhãn dạng số, shape (n_samples,)
                     Ví dụ: [0, 0, 1, 2, 2, ...] — 0=apple, 1=banana, ...
            verbose: In thông tin quá trình train

        Returns:
            dict chứa best_params, best_score và thời gian train
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC

        n_combos = len(self.c_values) * len(self.gamma_values)
        if verbose:
            print(f"Bắt đầu GridSearchCV với {n_combos} tổ hợp hyperparameter...")
            print(f"  C values   : {self.c_values}")
            print(f"  Gamma      : {self.gamma_values}")
            print(f"  CV folds   : {self.cv_folds}")
            print(f"  Tổng lần train: {n_combos * self.cv_folds}")
            print(f"  Dataset    : {X_train.shape[0]} ảnh, {X_train.shape[1]} đặc trưng")
            print("(Có thể mất 30 phút đến vài giờ tùy kích thước dataset và máy)\n")

        # CalibratedClassifierCV bọc ngoài SVC để có predict_proba() mà không cần
        # probability=True trong SVC (tham số đó đã deprecated từ sklearn 1.9).
        # ensemble=False → dùng cross-validation nội bộ để hiệu chỉnh xác suất.
        # Tham số C và gamma được truyền vào SVC bên trong thông qua param_grid.
        param_grid = {
            "estimator__C":     self.c_values,
            "estimator__gamma": self.gamma_values,
        }
        svc = SVC(kernel="rbf", random_state=42)
        calibrated_svc = CalibratedClassifierCV(svc, ensemble=False)
        grid_search = GridSearchCV(
            calibrated_svc,
            param_grid,
            cv=self.cv_folds,
            scoring="f1_macro",   # dùng F1 macro vì 36 class (không chỉ accuracy)
            n_jobs=-1,            # dùng hết CPU để chạy nhanh hơn
            verbose=2 if verbose else 0,
        )

        start = time.time()
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start

        self.model = grid_search.best_estimator_
        # best_params_ trả về key 'estimator__C' → đổi lại thành 'C' cho dễ đọc
        raw_params = grid_search.best_params_
        self.best_params = {
            "C":     raw_params["estimator__C"],
            "gamma": raw_params["estimator__gamma"],
        }
        self.is_fitted = True

        if verbose:
            print(f"\nTrain xong! Thời gian: {elapsed / 60:.1f} phút")
            print(f"Hyperparameter tốt nhất: C={self.best_params['C']}, gamma={self.best_params['gamma']}")
            print(f"F1-score tốt nhất (CV): {grid_search.best_score_:.4f}")

        return {
            "best_params": self.best_params,
            "best_cv_score": float(grid_search.best_score_),
            "train_time_seconds": elapsed,
            "train_time_minutes": elapsed / 60,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán class cho ảnh mới.

        Args:
            X: Ma trận đặc trưng đã qua PCA, shape (n_samples, n_components)

        Returns:
            Array nhãn dự đoán, shape (n_samples,)
        """
        self._check_fitted("predict()")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất cho từng class (dùng để hiển thị confidence score).

        Args:
            X: Ma trận đặc trưng đã qua PCA, shape (n_samples, n_components)

        Returns:
            Ma trận xác suất shape (n_samples, n_classes)
            Mỗi hàng là xác suất cho 36 class, tổng = 1.0
        """
        self._check_fitted("predict_proba()")
        return self.model.predict_proba(X)

    def predict_single(self, feature_vector: np.ndarray, top_k: int = 3) -> list:
        """
        Dự đoán 1 ảnh và trả về Top-K class có xác suất cao nhất.
        Dùng trong Streamlit để hiển thị kết quả.

        Args:
            feature_vector: Vector 1D shape (n_components,) — đặc trưng 1 ảnh sau PCA
            top_k: Trả về bao nhiêu class cao nhất

        Returns:
            List of tuples: [(class_index, class_name, probability), ...]
            Ví dụ: [(5, 'Carrot', 0.92), (12, 'Paprika', 0.05), ...]
        """
        self._check_fitted("predict_single()")
        proba = self.predict_proba(feature_vector.reshape(1, -1))[0]
        top_indices = np.argsort(proba)[::-1][:top_k]
        if self.classes is not None:
            return [(int(i), str(self.classes[i]), float(proba[i])) for i in top_indices]
        return [(int(i), f"class_{i}", float(proba[i])) for i in top_indices]

    def save(self, path: str) -> None:
        """
        Lưu model đã train ra file để dùng lại sau.

        Args:
            path: Đường dẫn lưu file, ví dụ "checkpoints/svm_model.pkl"
        """
        self._check_fitted("save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Đã lưu SVM model tại: {path}")

    @classmethod
    def load(cls, path: str) -> "ClassicalSVMClassifier":
        """
        Load model đã lưu từ file.

        Args:
            path: Đường dẫn file .pkl

        Returns:
            ClassicalSVMClassifier object đã train sẵn
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"Không tìm thấy file model: {path}")
        model = joblib.load(path)
        print(f"Đã load SVM model từ: {path}")
        return model


# ============================================================
# Hàm tiện ích: load dataset từ thư mục
# ============================================================

def _load_image_paths_and_labels(split_dir: str):
    """
    Đọc danh sách đường dẫn ảnh và nhãn từ thư mục có cấu trúc:
        split_dir/
            apple/   img1.jpg  img2.jpg ...
            banana/  img1.jpg ...
            ...

    Returns:
        (paths, labels, class_names):
            paths      — list đường dẫn ảnh
            labels     — list tên class tương ứng (string)
            class_names — list tên class đã sắp xếp (dùng để encode)
    """
    split_path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {split_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục class nào trong: {split_dir}")

    class_names = [d.name for d in class_dirs]
    paths, labels = [], []
    for cls_dir in class_dirs:
        files = [str(f) for f in cls_dir.iterdir() if f.suffix.lower() in exts]
        paths.extend(files)
        labels.extend([cls_dir.name] * len(files))

    print(f"  Tìm thấy {len(class_names)} class, {len(paths)} ảnh")
    return paths, labels, class_names


def _encode_labels(labels: list, class_names: list) -> np.ndarray:
    """Đổi nhãn string → số nguyên theo thứ tự class_names."""
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    return np.array([name_to_idx[lb] for lb in labels])


# ============================================================
# Hàm chạy toàn bộ pipeline
# ============================================================

def run_full_pipeline(
    data_dir: str = "data/raw",
    checkpoint_dir: str = "checkpoints",
    report_dir: str = "reports",
    pca_n_components: int = None,
    c_values: list = None,
    gamma_values: list = None,
    n_jobs_extract: int = 1,
):
    """
    Hàm chạy toàn bộ pipeline từ đầu đến cuối:
        Load ảnh → Trích HOG → PCA → Train SVM → Đánh giá → Lưu kết quả

    TN 3 gọi hàm này để train với toàn bộ dataset:
        python -m src.models.classical_svm --data-dir data/raw

    Args:
        data_dir       : Thư mục chứa data (có train/ validation/ test/ bên trong)
        checkpoint_dir : Nơi lưu model sau khi train
        report_dir     : Nơi lưu kết quả đánh giá
        pca_n_components: Số chiều PCA (None → lấy từ config, mặc định 200)
        c_values       : Ghi đè C list (None → lấy từ config)
        gamma_values   : Ghi đè gamma list (None → lấy từ config)
        n_jobs_extract : Số tiến trình trích đặc trưng song song (1 = tuần tự)
    """
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    from src.features.hog_features import batch_extract_features
    from src.features.pca_wrapper import PCAWrapper

    # Lấy config
    try:
        from configs.config import CONFIG
        svm_cfg = CONFIG["svm"]
        if pca_n_components is None:
            pca_n_components = svm_cfg["pca_n_components"]
        img_size = tuple(svm_cfg["img_size"])
    except (ImportError, KeyError):
        if pca_n_components is None:
            pca_n_components = 200
        img_size = (128, 128)

    data_dir = Path(data_dir)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Bước 1: Load danh sách ảnh
    # ----------------------------------------------------------------
    print("=" * 60)
    print("BƯỚC 1: Load dataset")
    print("=" * 60)
    train_paths, train_labels, class_names = _load_image_paths_and_labels(str(data_dir / "train"))
    val_paths,   val_labels,   _           = _load_image_paths_and_labels(str(data_dir / "validation"))
    test_paths,  test_labels,  _           = _load_image_paths_and_labels(str(data_dir / "test"))

    y_train = _encode_labels(train_labels, class_names)
    y_val   = _encode_labels(val_labels,   class_names)
    y_test  = _encode_labels(test_labels,  class_names)

    # ----------------------------------------------------------------
    # Bước 2: Trích đặc trưng HOG + Color Histogram
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 2: Trích đặc trưng HOG + Color Histogram")
    print("=" * 60)

    t0 = time.time()
    print("  Train set...")
    X_train_raw = batch_extract_features(train_paths, img_size=img_size,
                                          show_progress=True, n_jobs=n_jobs_extract)
    print("  Validation set...")
    X_val_raw   = batch_extract_features(val_paths,   img_size=img_size,
                                          show_progress=True, n_jobs=n_jobs_extract)
    print("  Test set...")
    X_test_raw  = batch_extract_features(test_paths,  img_size=img_size,
                                          show_progress=True, n_jobs=n_jobs_extract)
    t_extract = time.time() - t0
    print(f"\n  Trích đặc trưng xong: {t_extract:.1f}s")
    print(f"  Shape train: {X_train_raw.shape}, val: {X_val_raw.shape}, test: {X_test_raw.shape}")

    # ----------------------------------------------------------------
    # Bước 3: PCA giảm chiều
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"BƯỚC 3: PCA giảm chiều ({X_train_raw.shape[1]} → {pca_n_components})")
    print("=" * 60)

    pca = PCAWrapper(n_components=pca_n_components)
    X_train_pca = pca.fit_transform(X_train_raw)   # fit chỉ trên train!
    X_val_pca   = pca.transform(X_val_raw)
    X_test_pca  = pca.transform(X_test_raw)

    pca_path = str(Path(checkpoint_dir) / "pca_model.pkl")
    pca.save(pca_path)

    # ----------------------------------------------------------------
    # Bước 4: Train SVM với GridSearchCV
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 4: Train SVM (GridSearchCV)")
    print("=" * 60)

    classifier = ClassicalSVMClassifier(c_values=c_values, gamma_values=gamma_values)
    classifier.classes = class_names

    train_result = classifier.train(X_train_pca, y_train, verbose=True)

    svm_path = str(Path(checkpoint_dir) / "svm_model.pkl")
    classifier.save(svm_path)

    # ----------------------------------------------------------------
    # Bước 5: Đánh giá trên test set
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 5: Đánh giá trên test set")
    print("=" * 60)

    y_pred = classifier.predict(X_test_pca)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")

    report_str = classification_report(y_test, y_pred, target_names=class_names)
    print(report_str)
    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test F1 Macro : {f1:.4f}")

    # ----------------------------------------------------------------
    # Bước 6: Lưu kết quả
    # ----------------------------------------------------------------
    results = {
        "model_name": "classical_svm",
        "best_params": train_result["best_params"],
        "best_cv_f1": train_result["best_cv_score"],
        "train_time_minutes": train_result["train_time_minutes"],
        "pca_n_components": pca.n_components_,
        "pca_variance_kept": pca.get_total_variance_kept(),
        "test_accuracy": acc,
        "test_f1_macro": f1,
        "n_classes": len(class_names),
        "class_names": class_names,
    }

    results_path = str(Path(report_dir) / "svm_evaluation.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    report_path = str(Path(report_dir) / "svm_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"\nKết quả đã lưu tại: {results_path}")
    print(f"Báo cáo chi tiết  : {report_path}")
    print(f"SVM model         : {svm_path}")
    print(f"PCA model         : {pca_path}")

    return results


# ============================================================
# Chạy trực tiếp từ dòng lệnh:
#   python -m src.models.classical_svm --data-dir data/raw
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SVM classifier cho phân loại rau củ")
    parser.add_argument("--data-dir",       default="data/raw",    help="Thư mục chứa dataset")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Nơi lưu model")
    parser.add_argument("--report-dir",     default="reports",     help="Nơi lưu kết quả đánh giá")
    parser.add_argument("--pca-components", type=int, default=None, help="Số chiều PCA (mặc định từ config)")
    parser.add_argument("--n-jobs",         type=int, default=1,   help="Số tiến trình trích đặc trưng song song")
    args = parser.parse_args()

    print("🥕 Vegetable Classifier — SVM Pipeline")
    print(f"  Dataset  : {args.data_dir}")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print()

    results = run_full_pipeline(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        report_dir=args.report_dir,
        pca_n_components=args.pca_components,
        n_jobs_extract=args.n_jobs,
    )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print(f"  Accuracy  : {results['test_accuracy']*100:.2f}%")
    print(f"  F1 Macro  : {results['test_f1_macro']:.4f}")
    print(f"  Best C    : {results['best_params']['C']}")
    print(f"  Best gamma: {results['best_params']['gamma']}")
    print(f"  Train time: {results['train_time_minutes']:.1f} phút")
    print("=" * 60)
