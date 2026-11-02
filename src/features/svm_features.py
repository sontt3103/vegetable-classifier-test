"""
src/features/svm_features.py
----------------------------
Bộ đặc trưng v2 cho SVM – kết quả của thí nghiệm scripts/svm_experiments.py (cấu hình E5).

So với bộ v1 (hog_features.py: HOG cell 8 + histogram RGB):
    - HOG thô hơn (cell 16x16): 1.764 chiều thay vì 8.100, ít nhiễu nền hơn, chất lượng tương đương.
    - Histogram HSV thay cho RGB: kênh Hue (sắc độ) ít phụ thuộc ánh sáng hơn -> đóng góp lớn nhất.
    - Histogram HSV theo vùng (4 góc phần tư + vùng giữa): biết màu nằm ở đâu, giảm ảnh hưởng của nền.
    - LBP (Local Binary Pattern): đặc trưng kết cấu bề mặt (vỏ sần, vân, hạt).

Mọi khối nhận cùng một ảnh RGB uint8 đã resize về CONFIG["svm"]["img_size"] (128x128) bằng
hog_features.load_and_resize, nên huấn luyện và dự đoán dùng đúng một cách tiền xử lý.

Khối "cnn" (hướng B): vector 1.280 chiều lấy từ lớp Global Average Pooling của MobileNetV2 đã
huấn luyện sẵn trên ImageNet (KHÔNG huấn luyện lại – chỉ dùng làm bộ trích đặc trưng), ảnh vào
224x224. Mô hình phân loại vẫn là PCA + SVM. Khối này cần TensorFlow và được tính theo lô.

Pipeline mô hình (build_pipeline):
    HOG   -> PCA(50) -> chuẩn hóa L2  ┐
    HSV   -> Hellinger -> chuẩn hóa L2 ├-> ghép -> SVC RBF
    vùng  -> Hellinger -> chuẩn hóa L2 │
    LBP   -> Hellinger -> chuẩn hóa L2 ┘
Chuẩn hóa L2 theo từng khối giúp các khối có vai trò ngang nhau mà không cần StandardScaler
(StandardScaler làm mất thông tin màu – xem reports/svm_tuning_log.csv).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from configs.config import CONFIG

_SVM_CFG = CONFIG["svm"]
IMG_SIZE = tuple(_SVM_CFG["img_size"])
DEFAULT_BLOCKS = list(_SVM_CFG.get("v2_blocks", ["hog16", "hsv", "hsv_grid", "lbp"]))
HOG_ORIENTATIONS = 9
HSV_BINS = (32, 16, 16)          # H, S, V cho histogram toàn ảnh
HSV_GRID_BINS = (16, 8, 8)       # H, S, V cho mỗi vùng
LBP_SETTINGS = ((8, 1), (16, 2))  # (số điểm lân cận P, bán kính R), dạng "uniform"
CNN_INPUT = 224                  # kích thước ảnh vào MobileNetV2
CNN_DIM = 1280                   # số chiều đầu ra Global Average Pooling của MobileNetV2
# Trọng số ImageNet (tự tải về ~/.keras/models lần đầu). Đặt biến môi trường SVM_CNN_WEIGHTS=none
# để dùng mạng chưa huấn luyện – CHỈ dùng khi kiểm thử phần mềm, không dùng để báo cáo kết quả.
CNN_WEIGHTS = None if os.environ.get("SVM_CNN_WEIGHTS", "imagenet").lower() == "none" else "imagenet"
HANDCRAFTED = ("hog8", "hog16", "rgb", "hsv", "hsv_grid", "lbp")


# ---------------------------------------------------------------------------
# Các khối đặc trưng
# ---------------------------------------------------------------------------
def _hog(gray: np.ndarray, cell: int) -> np.ndarray:
    return hog(gray, orientations=HOG_ORIENTATIONS, pixels_per_cell=(cell, cell),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def _hist(channel: np.ndarray, bins: int, value_range: list) -> np.ndarray:
    h = cv2.calcHist([np.ascontiguousarray(channel)], [0], None, [bins], value_range).ravel()
    return (h / max(h.sum(), 1.0)).astype(np.float32)


def _hsv_hist(hsv: np.ndarray, bins=HSV_BINS) -> np.ndarray:
    # OpenCV: H trong [0, 180), S và V trong [0, 256)
    return np.concatenate([_hist(hsv[..., 0], bins[0], [0, 180]),
                           _hist(hsv[..., 1], bins[1], [0, 256]),
                           _hist(hsv[..., 2], bins[2], [0, 256])])


def _regions(img: np.ndarray) -> list[np.ndarray]:
    """4 góc phần tư + vùng giữa (1/2 cạnh), thường là nơi có vật thể chính."""
    h, w = img.shape[:2]
    return [img[:h // 2, :w // 2], img[:h // 2, w // 2:], img[h // 2:, :w // 2], img[h // 2:, w // 2:],
            img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]]


def _lbp(gray: np.ndarray) -> np.ndarray:
    out = []
    for p, r in LBP_SETTINGS:
        codes = local_binary_pattern(gray, p, r, method="uniform")
        h, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2))
        out.append(h / max(h.sum(), 1))
    return np.concatenate(out).astype(np.float32)


def extract_blocks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Ảnh RGB uint8 (đã resize) -> dict {tên khối: vector}. Có cả 2 kiểu HOG để làm thí nghiệm."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Cần ảnh RGB 3 kênh, nhận được shape {rgb.shape}")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "hog8": _hog(gray, 8),
        "hog16": _hog(gray, 16),
        "rgb": np.concatenate([_hist(rgb[..., c], 32, [0, 256]) for c in range(3)]),
        "hsv": _hsv_hist(hsv),
        "hsv_grid": np.concatenate([_hsv_hist(r, HSV_GRID_BINS) for r in _regions(hsv)]),
        "lbp": _lbp(gray),
    }


def extract_vector(rgb: np.ndarray, blocks: list[str] | None = None) -> np.ndarray:
    """Ảnh RGB đã resize về IMG_SIZE -> vector các khối THỦ CÔNG theo thứ tự `blocks`."""
    blocks = blocks or DEFAULT_BLOCKS
    if "cnn" in blocks:
        raise ValueError("Khối 'cnn' cần ảnh 224x224 – dùng extract_vector_from_image().")
    feats = extract_blocks(rgb)
    return np.concatenate([feats[b] for b in blocks])


def extract_vector_from_image(rgb_full: np.ndarray, blocks: list[str] | None = None) -> np.ndarray:
    """Ảnh RGB uint8 kích thước bất kỳ -> vector đầy đủ (kể cả khối 'cnn'), dùng khi dự đoán 1 ảnh.
    Resize giống hệt lúc huấn luyện: 128x128 cho khối thủ công, 224x224 cho khối cnn (INTER_AREA)."""
    blocks = blocks or DEFAULT_BLOCKS
    parts = {}
    hand = [b for b in blocks if b != "cnn"]
    if hand:
        small = cv2.resize(rgb_full, IMG_SIZE, interpolation=cv2.INTER_AREA)
        feats = extract_blocks(small)
        parts.update({b: feats[b] for b in hand})
    if "cnn" in blocks:
        big = cv2.resize(rgb_full, (CNN_INPUT, CNN_INPUT), interpolation=cv2.INTER_AREA)
        parts["cnn"] = extract_cnn_batch(big[None])[0]
    return np.concatenate([parts[b] for b in blocks])


def block_dims(blocks: list[str] | None = None) -> dict[str, int]:
    blocks = blocks or DEFAULT_BLOCKS
    dummy = np.zeros((*IMG_SIZE[::-1], 3), dtype=np.uint8)
    feats = extract_blocks(dummy)
    return {b: (CNN_DIM if b == "cnn" else int(feats[b].shape[0])) for b in blocks}


def feature_params(blocks: list[str] | None = None) -> dict:
    """Tham số đặc trưng lưu kèm mô hình; svm_predict so sánh để tránh lệch train/dự đoán."""
    params = {
        "feature_set": "v2",
        "img_size": list(IMG_SIZE),
        "blocks": list(blocks or DEFAULT_BLOCKS),
        "hog_orientations": HOG_ORIENTATIONS,
        "hsv_bins": list(HSV_BINS),
        "hsv_grid_bins": list(HSV_GRID_BINS),
        "lbp_settings": [list(s) for s in LBP_SETTINGS],
    }
    if "cnn" in params["blocks"]:
        params["cnn_backbone"] = f"MobileNetV2-{CNN_WEIGHTS}-avgpool"
        params["cnn_input"] = CNN_INPUT
    return params


# ---------------------------------------------------------------------------
# Khối "cnn": MobileNetV2 đã huấn luyện sẵn làm bộ trích đặc trưng (không fine-tune)
# ---------------------------------------------------------------------------
_CNN_MODEL = None


def _cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is None:
        import tensorflow as tf
        _CNN_MODEL = tf.keras.applications.MobileNetV2(
            include_top=False, pooling="avg", weights=CNN_WEIGHTS, input_shape=(CNN_INPUT, CNN_INPUT, 3))
        _CNN_MODEL.trainable = False
    return _CNN_MODEL


def extract_cnn_batch(images: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """images: mảng uint8 (N, 224, 224, 3) RGB -> đặc trưng (N, 1280).
    Chuẩn hóa đúng chuẩn đầu vào MobileNetV2: giá trị điểm ảnh về [-1, 1]."""
    x = images.astype(np.float32) / 127.5 - 1.0
    return _cnn_model().predict(x, batch_size=batch_size, verbose=0).astype(np.float32)


def _load_rgb(path: str, size: tuple):
    from src.features.hog_features import load_and_resize
    try:
        return load_and_resize(path, size)
    except Exception:  # noqa: BLE001
        return None


def _cnn_for_paths(paths: list[str], with_flip: bool, n_jobs: int, chunk: int = 256):
    from joblib import Parallel, delayed
    feats, feats_flip, keep = [], [], []
    for i in range(0, len(paths), chunk):
        imgs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load_rgb)(p, (CNN_INPUT, CNN_INPUT)) for p in paths[i:i + chunk])
        keep += [im is not None for im in imgs]
        ok = np.stack([im for im in imgs if im is not None]) if any(im is not None for im in imgs) else None
        if ok is None:
            continue
        feats.append(extract_cnn_batch(ok))
        if with_flip:
            feats_flip.append(extract_cnn_batch(np.ascontiguousarray(ok[:, :, ::-1])))
    X = np.concatenate(feats) if feats else np.zeros((0, CNN_DIM), np.float32)
    Xf = np.concatenate(feats_flip) if with_flip and feats_flip else None
    return X, Xf, np.array(keep, dtype=bool)


# ---------------------------------------------------------------------------
# Trích đặc trưng hàng loạt (song song), kèm bản lật ngang để tăng dữ liệu
# ---------------------------------------------------------------------------
def _extract_path(path: str, blocks: list[str], with_flip: bool):
    from src.features.hog_features import load_and_resize
    try:
        rgb = load_and_resize(path, IMG_SIZE)
    except Exception:  # noqa: BLE001 – ảnh lỗi được bỏ qua và báo lại
        return None
    x = extract_vector(rgb, blocks)
    xf = extract_vector(np.ascontiguousarray(rgb[:, ::-1]), blocks) if with_flip else None
    return x, xf


def batch_extract(paths: list[str], blocks: list[str] | None = None, with_flip: bool = False,
                  n_jobs: int = -1):
    """Trả về (X, X_flip hoặc None, keep_mask)."""
    from joblib import Parallel, delayed

    blocks = blocks or DEFAULT_BLOCKS
    hand = [b for b in blocks if b != "cnn"]
    if hand and blocks[blocks.index(hand[0]): blocks.index(hand[0]) + len(hand)] != hand:
        raise ValueError("Các khối thủ công phải đứng liền nhau trong danh sách blocks (vd ['cnn', 'hsv', 'lbp']).")

    keep = np.ones(len(paths), dtype=bool)
    hand_res, cnn_res = {}, {}
    if hand:
        res = Parallel(n_jobs=n_jobs)(delayed(_extract_path)(p, hand, with_flip) for p in paths)
        keep &= np.array([r is not None for r in res], dtype=bool)
        hand_res = {i: r for i, r in enumerate(res) if r is not None}
    if "cnn" in blocks:
        Xc, Xcf, keep_c = _cnn_for_paths(paths, with_flip, n_jobs)
        cnn_res = {i: (Xc[j], Xcf[j] if with_flip else None) for j, i in enumerate(np.flatnonzero(keep_c))}
        keep &= keep_c
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        raise ValueError("Không trích được đặc trưng từ ảnh nào.")

    def _assemble(i: int, k: int) -> np.ndarray:   # k = 0: ảnh gốc, k = 1: ảnh lật
        out = []
        for b in blocks:
            if b == "cnn":
                out.append(cnn_res[i][k])
            elif b == hand[0]:            # cả nhóm khối thủ công (liền nhau) đã được ghép sẵn
                out.append(hand_res[i][k])
        return np.concatenate(out)

    X = np.stack([_assemble(i, 0) for i in idx]).astype(np.float32)
    X_flip = np.stack([_assemble(i, 1) for i in idx]).astype(np.float32) if with_flip else None
    return X, X_flip, keep


# ---------------------------------------------------------------------------
# Pipeline mô hình
# ---------------------------------------------------------------------------
def hellinger(x):
    """Biến đổi Hellinger: căn bậc hai tỉ lệ pixel -> bin nhỏ không bị bin lớn (màu nền) lấn át.
    Đặt ở mức module để pickle được (Streamlit nạp lại mô hình)."""
    return np.sqrt(np.clip(x, 0, None))


def build_pipeline(blocks: list[str], dims: dict[str, int], hog_pca: int = 50, seed: int = 42,
                   cnn_pca: int = 256):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, Normalizer
    from sklearn.svm import SVC

    parts, start = [], 0
    for b in blocks:
        cols = list(range(start, start + dims[b]))
        start += dims[b]
        if b.startswith("hog"):
            step = Pipeline([("pca", PCA(hog_pca, random_state=seed)), ("l2", Normalizer())])
        elif b == "cnn":   # căn bậc hai (power normalization) -> L2 -> PCA -> L2
            step = Pipeline([("sqrt", FunctionTransformer(hellinger)), ("l2", Normalizer()),
                             ("pca", PCA(cnn_pca, random_state=seed)), ("l2b", Normalizer())])
        else:
            step = Pipeline([("hellinger", FunctionTransformer(hellinger)), ("l2", Normalizer())])
        parts.append((b, step, cols))
    return Pipeline([("features", ColumnTransformer(parts)),
                     ("svc", SVC(kernel="rbf", gamma="scale", random_state=seed))])


BLOCK_NAMES = {"hog8": "HOG (cell 8)", "hog16": "HOG (cell 16)", "rgb": "Histogram RGB", "hsv": "Histogram HSV",
               "hsv_grid": "HSV theo vùng", "lbp": "LBP", "cnn": "Đặc trưng MobileNetV2 (ImageNet)"}


def describe_blocks(blocks: list[str], best_params: dict | None = None) -> str:
    """Mô tả ngắn gọn bộ đặc trưng, dùng cho metadata và giao diện."""
    bp = best_params or {}
    out = []
    for b in blocks:
        name = BLOCK_NAMES.get(b, b)
        if b.startswith("hog") and bp.get("hog_pca"):
            name += f" -> PCA({bp['hog_pca']})"
        if b == "cnn" and bp.get("cnn_pca"):
            name += f" -> PCA({bp['cnn_pca']})"
        out.append(name)
    return " + ".join(out)


def model_dims(blocks: list[str], dims: dict[str, int], hog_pca: int, cnn_pca: int = 256) -> int:
    """Số chiều thực sự đưa vào SVC (HOG, CNN sau PCA + các histogram)."""
    return sum(hog_pca if b.startswith("hog") else cnn_pca if b == "cnn" else dims[b] for b in blocks)
