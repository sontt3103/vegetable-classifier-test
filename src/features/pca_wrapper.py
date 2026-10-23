"""
src/features/pca_wrapper.py
----------------------------
NGƯỜI LÀM: Nguyen Duc Tai

Mục đích:
    Giảm số chiều của vector đặc trưng trước khi đưa vào SVM.

    Vấn đề: HOG + Color Histogram tạo ra 8196 số cho mỗi ảnh (xem hog_features.py).
    SVM train chậm hơn khi vector quá dài. PCA giúp giảm từ ~8000 xuống còn ~200
    mà vẫn giữ lại phần lớn thông tin quan trọng.

    Nghĩ đơn giản: PCA tìm cách "nén" dữ liệu mà không mất nhiều thông tin.

Cách dùng:
    from src.features.pca_wrapper import PCAWrapper

    pca = PCAWrapper(n_components=200)
    X_train_reduced = pca.fit_transform(X_train)  # Train PCA trên train set
    X_test_reduced = pca.transform(X_test)        # Chỉ transform, không fit lại!
    pca.save("checkpoints/pca_model.pkl")         # Lưu để dùng khi inference

    # Sau đó load lại:
    pca2 = PCAWrapper.load("checkpoints/pca_model.pkl")

Các cách đặt n_components:
    200      → giữ đúng 200 chiều (số nguyên)
    0.95     → tự chọn số chiều ít nhất để giữ >= 95% phương sai (số thực trong (0, 1))
    None     → tương đương 0.95
"""

import warnings
from pathlib import Path

import joblib
import numpy as np

# Số chiều mặc định khi n_components=None
DEFAULT_VARIANCE_TARGET = 0.95


class PCAWrapper:
    """
    Bọc sklearn.decomposition.PCA với các tính năng thêm:
    - Tự ghi lại % phương sai giữ lại
    - Có hàm save/load model
    - Có thể vẽ biểu đồ explained variance (dùng trong EDA)
    """

    def __init__(self, n_components=200, standardize: bool = False, random_state: int = 42):
        """
        Args:
            n_components: Số chiều sau khi giảm.
                          - int: giữ đúng số chiều đó (vd 200). Thường dùng 100-300.
                          - float trong (0, 1): giữ số chiều ít nhất đủ đạt tỉ lệ phương sai đó.
                          - None: giữ 95% phương sai.
                          (LƯU Ý: sklearn PCA(n_components=None) mặc định giữ TẤT CẢ
                          component, không phải 95% — wrapper này đã xử lý lại cho đúng.)
            standardize: True → chuẩn hoá từng feature về mean=0, std=1 trước khi PCA.
                         Mặc định False (PCA chỉ trừ mean). Có thể thử True để so sánh,
                         nhưng HOG ở vùng ảnh phẳng có phương sai rất nhỏ → chuẩn hoá
                         có thể khuếch đại nhiễu.
            random_state: Seed để kết quả PCA tái lập được.
        """
        self.n_components = n_components
        self.standardize = standardize
        self.random_state = random_state

        self.pca = None          # Sẽ là sklearn PCA object sau khi fit
        self.scaler = None       # StandardScaler nếu standardize=True
        self.is_fitted = False   # Cờ kiểm tra đã fit chưa
        self.n_features_in_ = None   # Số chiều đầu vào (đặt khi fit)
        self.n_components_ = None    # Số chiều đầu ra THỰC TẾ (đặt khi fit)

    # ------------------------------------------------------------------
    # Kiểm tra / tiện ích nội bộ
    # ------------------------------------------------------------------
    def _check_fitted(self, action: str = "dùng") -> None:
        if not self.is_fitted:
            raise RuntimeError(f"Cần gọi fit() trước khi {action}")

    @staticmethod
    def _validate_matrix(X) -> np.ndarray:
        """Đảm bảo X là ma trận 2D hữu hạn (không NaN/Inf)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"X phải là ma trận 2D (n_samples, n_features), nhận được shape {X.shape}. "
                "Nếu chỉ có 1 ảnh, dùng X.reshape(1, -1)."
            )
        if not np.isfinite(X).all():
            raise ValueError("X chứa NaN hoặc Inf — kiểm tra lại bước rút trích đặc trưng.")
        return X

    def _resolve_n_components(self, n_samples: int, n_features: int):
        """
        Đổi self.n_components thành giá trị hợp lệ cho sklearn.
        PCA yêu cầu số chiều <= min(n_samples, n_features) — nếu test với ít ảnh
        (vd 10-20 ảnh) mà đòi 200 chiều thì sklearn báo lỗi, nên ở đây tự hạ xuống.
        """
        nc = DEFAULT_VARIANCE_TARGET if self.n_components is None else self.n_components

        if isinstance(nc, (float, np.floating)):
            if not 0.0 < nc < 1.0:
                raise ValueError(f"n_components kiểu float phải nằm trong (0, 1), nhận được {nc}")
            return float(nc), "full"   # chế độ chọn theo % phương sai cần svd_solver='full'

        if isinstance(nc, (int, np.integer)):
            if nc < 1:
                raise ValueError(f"n_components phải >= 1, nhận được {nc}")
            max_allowed = min(n_samples, n_features)
            if nc > max_allowed:
                warnings.warn(
                    f"n_components={nc} lớn hơn giới hạn min(n_samples, n_features)={max_allowed}; "
                    f"tự giảm xuống {max_allowed}."
                )
                nc = max_allowed
            return int(nc), "auto"

        raise TypeError(f"n_components phải là int, float hoặc None, nhận được {type(nc)}")

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "PCAWrapper":
        """
        Học PCA từ dữ liệu train.

        QUAN TRỌNG: Chỉ gọi fit() trên train set!
        Không được fit trên validation hoặc test set → gây data leakage.

        Args:
            X: Ma trận đặc trưng shape (n_samples, n_features)
               Ví dụ: (5000, 8196) — 5000 ảnh, mỗi ảnh 8196 đặc trưng

        Returns:
            self (để có thể chain: pca.fit(X).transform(X))
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = self._validate_matrix(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("Cần ít nhất 2 mẫu để fit PCA")

        n_comp, solver = self._resolve_n_components(n_samples, n_features)

        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        self.pca = PCA(n_components=n_comp, svd_solver=solver, random_state=self.random_state)
        self.pca.fit(X)

        self.n_features_in_ = n_features
        self.n_components_ = int(self.pca.n_components_)
        self.is_fitted = True

        variance_kept = float(np.sum(self.pca.explained_variance_ratio_)) * 100
        print(f"PCA fit xong: {n_features} chiều → {self.n_components_} chiều")
        print(f"Phương sai giữ lại: {variance_kept:.1f}%")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Áp dụng PCA đã học để giảm chiều dữ liệu mới.

        Args:
            X: Ma trận shape (n_samples, n_features_original)

        Returns:
            Ma trận shape (n_samples, n_components) — ít cột hơn
        """
        self._check_fitted("transform()")
        X = self._validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Số chiều đầu vào ({X.shape[1]}) khác số chiều lúc fit ({self.n_features_in_}). "
                "Thường do dùng img_size / tham số HOG khác với lúc train."
            )
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Gọi fit() rồi transform() trong 1 bước (tiện hơn).
        Chỉ dùng cho train set!

        Args:
            X: Ma trận shape (n_samples, n_features)

        Returns:
            Ma trận shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Trả về % phương sai mà mỗi component giải thích.
        Dùng để vẽ biểu đồ trong EDA notebook.

        Returns:
            Array shape (n_components,), ví dụ [0.15, 0.08, 0.05, ...]
            Nghĩa là: component đầu tiên giữ 15% thông tin, thứ 2 giữ 8%, ...
        """
        self._check_fitted("lấy explained variance")
        return self.pca.explained_variance_ratio_

    def get_total_variance_kept(self) -> float:
        """Tổng tỉ lệ phương sai (0-1) mà toàn bộ các component đang giữ lại."""
        return float(np.sum(self.get_explained_variance_ratio()))

    def n_components_for_variance(self, threshold: float = 0.95):
        """
        Số component ít nhất để giữ >= threshold phương sai (vd 0.95 → 95%).

        Returns:
            int, hoặc None nếu PCA này (đã giữ ít component) không đạt tới ngưỡng đó.
            Muốn có đường cong đầy đủ, fit PCA với nhiều component hơn (vd EDA).
        """
        cumsum = np.cumsum(self.get_explained_variance_ratio())
        if cumsum[-1] < threshold:
            return None
        return int(np.searchsorted(cumsum, threshold) + 1)

    def save(self, path: str) -> None:
        """
        Lưu PCA model ra file để dùng lại sau (inference, so sánh).

        Lưu dạng dict các thành phần (scaler + sklearn PCA) thay vì pickle cả object,
        để file vẫn load được dù sau này đổi vị trí/tên class (và Streamlit không bị
        phụ thuộc vào việc import đúng đường dẫn src.features...).

        Args:
            path: Đường dẫn lưu file, ví dụ "checkpoints/pca_model.pkl"
        """
        self._check_fitted("save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "n_components": self.n_components,
                "standardize": self.standardize,
                "random_state": self.random_state,
                "pca": self.pca,
                "scaler": self.scaler,
                "n_features_in_": self.n_features_in_,
                "n_components_": self.n_components_,
            },
            path,
        )
        print(f"Đã lưu PCA model tại: {path}")

    @classmethod
    def load(cls, path: str) -> "PCAWrapper":
        """
        Load PCA model đã lưu từ file.

        Args:
            path: Đường dẫn file .pkl đã lưu

        Returns:
            PCAWrapper object đã fit sẵn, dùng được ngay
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"Không tìm thấy file PCA model: {path}")
        state = joblib.load(path)

        model = cls(
            n_components=state["n_components"],
            standardize=state["standardize"],
            random_state=state["random_state"],
        )
        model.pca = state["pca"]
        model.scaler = state["scaler"]
        model.n_features_in_ = state["n_features_in_"]
        model.n_components_ = state["n_components_"]
        model.is_fitted = True
        print(f"Đã load PCA model từ: {path}")
        return model

    def plot_explained_variance(self, save_path: str = None, show: bool = False):
        """
        Vẽ biểu đồ % phương sai tích lũy theo số component.
        Dùng trong EDA notebook để chọn n_components phù hợp.

        Ví dụ biểu đồ cho thấy: 200 component giữ 95% phương sai
        → chọn n_components=200 là hợp lý.

        Args:
            save_path: Nếu truyền vào, lưu ảnh ra file đó (dùng trong báo cáo)
            show: True → gọi plt.show() (dùng khi chạy script; trong notebook
                  cứ để False rồi gọi plt.show() ở cell)

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        cumsum = np.cumsum(self.get_explained_variance_ratio()) * 100
        x = np.arange(1, len(cumsum) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, cumsum, color="#4C9A6B", linewidth=2, label="Phương sai tích lũy")
        ax.axhline(y=95, color="r", linestyle="--", label="Ngưỡng 95%")

        n95 = self.n_components_for_variance(0.95)
        if n95 is not None:
            ax.axvline(x=n95, color="gray", linestyle=":")
            ax.annotate(
                f"{n95} component\n→ 95%",
                xy=(n95, 95), xytext=(n95, max(cumsum.min(), 95 - 25)),
                ha="center", fontsize=9,
            )

        ax.set_xlabel("Số PCA components")
        ax.set_ylabel("% Phương sai tích lũy")
        ax.set_title("PCA: Số component cần thiết để giữ bao nhiêu thông tin")
        ax.set_ylim(top=100.5)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig


# ============================================================
# TEST NHANH
# python src/features/pca_wrapper.py
# ============================================================
if __name__ == "__main__":
    import tempfile

    print("=== Test PCA Wrapper ===")

    # Dữ liệu giả có cấu trúc (không phải nhiễu thuần) để PCA có gì đó để nén
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(100, 10))
    X_fake = latent @ rng.normal(size=(10, 8196)) + 0.05 * rng.normal(size=(100, 8196))

    pca = PCAWrapper(n_components=50)
    X_reduced = pca.fit_transform(X_fake)
    print("Shape sau PCA:", X_reduced.shape)
    assert X_reduced.shape == (100, 50), "Sai shape!"

    # Save / load phải cho kết quả giống hệt
    with tempfile.TemporaryDirectory() as tmp:
        p = f"{tmp}/pca_test.pkl"
        pca.save(p)
        pca2 = PCAWrapper.load(p)
        assert np.allclose(pca2.transform(X_fake), X_reduced), "Load xong kết quả khác!"

    # Chế độ chọn theo % phương sai
    pca95 = PCAWrapper(n_components=0.95).fit(X_fake)
    print("Chế độ 95% phương sai →", pca95.n_components_, "component")

    print("OK — PCA hoạt động đúng.")