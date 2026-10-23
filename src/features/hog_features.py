"""
src/features/hog_features.py
----------------------------
NGƯỜI LÀM: Nguyen Duc Tai

Mục đích:
    Rút trích đặc trưng từ ảnh bằng 2 phương pháp:
    - HOG (Histogram of Oriented Gradients): phát hiện đường viền và hình dạng trong ảnh
    - Color Histogram: phân bố màu sắc của ảnh

    Đặc trưng = thông tin dạng số mà SVM có thể học được.
    Ví dụ: ảnh cà rốt có màu cam và hình dài → vector số biểu diễn điều đó.

Cách dùng:
    from src.features.hog_features import extract_combined_features
    features = extract_combined_features("path/to/carrot.jpg")
    # features là 1 mảng numpy 1D, với tham số mặc định có 8196 số:
    #   HOG 8100 số + Color histogram 96 số

Số chiều đặc trưng (với tham số mặc định) — TN 1 / TN 3 cần biết:
    Ảnh 128x128, cell 8x8  → lưới 16x16 cell
    Block 2x2 cell, bước 1 → (16-2+1) x (16-2+1) = 15 x 15 = 225 block
    Mỗi block = 2*2 cell * 9 hướng = 36 số   → HOG = 225 * 36 = 8100 số
    Color hist = 32 bin * 3 kênh              → 96 số
    TỔNG = 8100 + 96 = 8196 số
    (Dùng get_feature_dim() để tính lại nếu đổi tham số, đừng hard-code.)

Trọng số màu (COLOR_WEIGHT) — QUAN TRỌNG:
    HOG có 8100 số, mỗi block đã được chuẩn hoá (norm = 1) nên tổng phương sai rất lớn.
    Histogram màu chỉ là tỉ lệ pixel (tổng mỗi kênh = 1) nên tổng phương sai rất nhỏ
    (chỉ ~1% so với HOG). PCA chọn các hướng có phương sai lớn nhất → nếu ghép thẳng,
    thông tin màu gần như BỊ PCA BỎ QUA, trong khi màu là đặc trưng rất mạnh để phân
    biệt rau củ (chuối vàng, cà chua đỏ...). Vì vậy phần màu được nhân với COLOR_WEIGHT
    (mặc định 10) trước khi ghép. Hệ số này chỉ phóng to thang đo, không đổi số chiều.
    Train và dự đoán PHẢI dùng cùng một giá trị — đã được đảm bảo vì cả hai cùng gọi
    extract_combined_features / extract_features_from_image. Có thể thử 1, 5, 10, 20
    khi tune SVM (đổi 1 hằng số này rồi trích lại đặc trưng + fit lại PCA).
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


# ============================================================
# CÁC THAM SỐ MẶC ĐỊNH — có thể thay đổi để thử nghiệm
# ============================================================
DEFAULT_IMG_SIZE = (128, 128)   # Resize ảnh về kích thước này trước khi xử lý (width, height)
HOG_ORIENTATIONS = 9            # Số hướng gradient (thường dùng 8 hoặc 9)
HOG_PIXELS_PER_CELL = (8, 8)   # Kích thước mỗi ô nhỏ trong HOG
HOG_CELLS_PER_BLOCK = (2, 2)   # Số ô ghép lại thành 1 block để normalize
COLOR_HIST_BINS = 32            # Số bin cho histogram màu (mỗi kênh RGB)
COLOR_WEIGHT = 10.0             # Hệ số nhân cho phần màu khi ghép với HOG (xem docstring đầu file)


def _read_image_rgb(image_path) -> np.ndarray:
    """
    Đọc file ảnh thành mảng RGB uint8 (H, W, 3).

    Dùng np.fromfile + cv2.imdecode thay cho cv2.imread vì cv2.imread KHÔNG đọc
    được đường dẫn có ký tự Unicode (ví dụ thư mục tiếng Việt có dấu) trên Windows
    — nó âm thầm trả về None. Nếu OpenCV không giải mã được (vd .webp lạ, ảnh
    palette/RGBA) thì thử lại bằng Pillow.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size > 0 else None

    if img is not None:
        # OpenCV đọc theo BGR → đổi sang RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dự phòng: Pillow (xử lý tốt ảnh grayscale / RGBA / palette / webp)
    try:
        from PIL import Image
        with Image.open(path) as pil_img:
            return np.array(pil_img.convert("RGB"), dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Không đọc được ảnh (file hỏng hoặc sai định dạng): {path}") from e


def load_and_resize(image_path: str, img_size: tuple = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn và resize về kích thước cố định.

    Tại sao cần resize?
        SVM cần vector đầu vào có cùng số chiều.
        Nếu ảnh 200x300 và ảnh 100x100 → vector khác chiều → SVM không chạy được.

    Args:
        image_path: Đường dẫn đến file ảnh (jpg, png, ...)
        img_size: Kích thước mục tiêu (width, height), mặc định (128, 128)

    Returns:
        Ảnh numpy array shape (128, 128, 3), dtype uint8, giá trị 0-255, thứ tự kênh RGB

    Raises:
        FileNotFoundError: file không tồn tại
        ValueError: file tồn tại nhưng không phải ảnh hợp lệ
    """
    img = _read_image_rgb(image_path)
    # INTER_AREA là phép nội suy tốt nhất khi THU NHỎ ảnh (giảm răng cưa)
    return cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_AREA)


def extract_hog_features(image: np.ndarray, visualize: bool = False):
    """
    Tính HOG (Histogram of Oriented Gradients) từ ảnh.

    HOG là gì?
        HOG chia ảnh thành các ô nhỏ, trong mỗi ô tính xem có
        bao nhiêu đường thẳng theo từng hướng (0°, 45°, 90°, ...).
        Ảnh cà rốt sẽ có nhiều đường dọc (thân dài), ảnh cam sẽ tròn hơn.

    Args:
        image: Ảnh numpy array shape (H, W, 3), giá trị 0-255 (RGB)
        visualize: Nếu True, trả thêm ảnh HOG để vẽ minh họa (dùng trong EDA)

    Returns:
        visualize=False (mặc định): vector 1D numpy array
            (8100 số với ảnh 128x128 và tham số mặc định)
        visualize=True: tuple (vector, hog_image) — hog_image là ảnh 2D để imshow
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:          # ảnh đã là grayscale
        gray = image
    else:
        raise ValueError(f"Ảnh phải có shape (H, W, 3) hoặc (H, W), nhận được {image.shape}")

    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,
    )


def extract_color_histogram(image: np.ndarray, bins: int = COLOR_HIST_BINS) -> np.ndarray:
    """
    Tính histogram màu sắc từ ảnh.

    Histogram màu là gì?
        Đếm xem có bao nhiêu pixel có màu đỏ nhạt, đỏ đậm, xanh lá nhạt, ...
        Ảnh ớt đỏ sẽ có nhiều pixel màu đỏ, ảnh rau cải sẽ có nhiều pixel xanh lá.

    Args:
        image: Ảnh numpy array shape (H, W, 3), RGB, giá trị 0-255
        bins: Số mức màu cho mỗi kênh (32 bins × 3 kênh = 96 số)

    Returns:
        Vector 1D numpy array (bins * 3 số). Mỗi kênh được chia cho tổng số pixel
        nên đó là TỈ LỆ pixel (tổng mỗi kênh = 1) → không phụ thuộc kích thước ảnh.
        Thứ tự: [R (bins số), G (bins số), B (bins số)]
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Ảnh phải có shape (H, W, 3), nhận được {image.shape}")

    image = np.ascontiguousarray(image, dtype=np.uint8)
    n_pixels = image.shape[0] * image.shape[1]

    hist_features = []
    for channel in range(3):  # R, G, B
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256]).flatten()
        hist_features.append(hist / n_pixels)
    return np.concatenate(hist_features)


def get_feature_dim(img_size: tuple = DEFAULT_IMG_SIZE, bins: int = COLOR_HIST_BINS) -> dict:
    """
    Tính số chiều đặc trưng mà KHÔNG cần đọc ảnh nào (chạy trên 1 ảnh trắng).

    Dùng để TN 1 / TN 3 / TN 4 lấy số chiều chính xác thay vì hard-code.

    Returns:
        dict {"hog": ..., "color": ..., "total": ...}
        Với tham số mặc định: {"hog": 8100, "color": 96, "total": 8196}
    """
    w, h = img_size
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    n_hog = extract_hog_features(dummy).shape[0]
    n_color = bins * 3
    return {"hog": int(n_hog), "color": int(n_color), "total": int(n_hog + n_color)}


def extract_features_from_image(image: np.ndarray, color_weight: float = COLOR_WEIGHT) -> np.ndarray:
    """
    Gộp HOG + Color Histogram từ 1 ảnh ĐÃ resize sẵn (mảng RGB, không phải đường dẫn).

    Tách riêng để Streamlit (ảnh user upload là bytes/PIL, không có đường dẫn file)
    và EDA notebook có thể dùng lại — đảm bảo ảnh lúc train và lúc dự đoán được
    xử lý GIỐNG HỆT nhau.

    Args:
        image: Ảnh RGB uint8 đã resize (H, W, 3)
        color_weight: Hệ số nhân phần màu (mặc định COLOR_WEIGHT)
    """
    hog_feat = extract_hog_features(image)
    color_feat = extract_color_histogram(image) * color_weight
    return np.concatenate([hog_feat, color_feat]).astype(np.float32)


def extract_combined_features(
    image_path: str,
    img_size: tuple = DEFAULT_IMG_SIZE,
    color_weight: float = COLOR_WEIGHT,
) -> np.ndarray:
    """
    Hàm CHÍNH — gộp HOG + Color Histogram thành 1 vector đặc trưng.

    Đây là hàm TN 1 và TN 3 sẽ gọi, không cần biết chi tiết bên trong.

    Args:
        image_path: Đường dẫn đến file ảnh
        img_size: Kích thước ảnh sau resize
        color_weight: Hệ số nhân phần màu (mặc định COLOR_WEIGHT = 10, xem docstring đầu file)

    Returns:
        Vector 1D numpy array (float32) gộp HOG + color hist
        Với tham số mặc định: HOG (8100 số) + Color hist (96 số) = 8196 số tổng cộng
    """
    image = load_and_resize(image_path, img_size)
    return extract_features_from_image(image, color_weight)


def _safe_extract(path, img_size, color_weight):
    """Bọc extract_combined_features: trả (vector, None) nếu OK, (None, lỗi) nếu hỏng."""
    try:
        return extract_combined_features(path, img_size, color_weight), None
    except Exception as e:  # noqa: BLE001
        return None, f"{path}: {e}"


def batch_extract_features(
    image_paths: list,
    img_size: tuple = DEFAULT_IMG_SIZE,
    show_progress: bool = True,
    on_error: str = "raise",
    return_valid_mask: bool = False,
    n_jobs: int = 1,
    color_weight: float = COLOR_WEIGHT,
):
    """
    Rút trích đặc trưng cho NHIỀU ảnh cùng lúc (dùng khi xử lý toàn bộ dataset).

    Args:
        image_paths: List các đường dẫn ảnh
        img_size: Kích thước ảnh sau resize
        show_progress: Hiện thanh tiến độ (tqdm nếu có, không thì in mỗi 100 ảnh)
        on_error: Xử lý ảnh hỏng:
            "raise" (mặc định) — dừng ngay và báo ảnh nào hỏng
            "skip"  — bỏ qua ảnh hỏng, in cảnh báo. CHÚ Ý: khi đó số hàng của kết quả
                      ít hơn len(image_paths) → nhãn y sẽ bị lệch nếu không lọc theo
                      mask. Luôn dùng kèm return_valid_mask=True và lọc y[mask].
        return_valid_mask: Nếu True, trả thêm mảng bool (n_images,) — True = ảnh đọc được
        n_jobs: Số tiến trình chạy song song (1 = tuần tự, -1 = dùng hết CPU).
                Song song giúp nhanh hơn nhiều khi xử lý toàn bộ dataset.
        color_weight: Hệ số nhân phần màu (mặc định COLOR_WEIGHT = 10)

    Returns:
        Ma trận 2D numpy array float32 shape (n_images, n_features)
        Ví dụ: 1000 ảnh → ma trận (1000, 8196)
        Nếu return_valid_mask=True: tuple (ma trận, valid_mask)
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error phải là 'raise' hoặc 'skip'")

    image_paths = list(image_paths)
    n_total = len(image_paths)
    if n_total == 0:
        raise ValueError("image_paths rỗng — không có ảnh nào để xử lý")

    # --- Chạy trích đặc trưng (tuần tự hoặc song song) ---
    if n_jobs == 1:
        results = []
        iterator = image_paths
        try:
            from tqdm import tqdm
            if show_progress:
                iterator = tqdm(image_paths, desc="Trích đặc trưng", unit="ảnh")
        except ImportError:
            pass
        for i, path in enumerate(iterator):
            results.append(_safe_extract(path, img_size, color_weight))
            if show_progress and iterator is image_paths and (i + 1) % 100 == 0:
                print(f"Đã xử lý {i + 1}/{n_total} ảnh")
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=10 if show_progress else 0)(
            delayed(_safe_extract)(p, img_size, color_weight) for p in image_paths
        )

    # --- Gom kết quả, xử lý ảnh hỏng ---
    errors = [err for _, err in results if err is not None]
    if errors and on_error == "raise":
        raise ValueError(
            f"{len(errors)}/{n_total} ảnh bị lỗi. Lỗi đầu tiên: {errors[0]}\n"
            "Dùng on_error='skip', return_valid_mask=True để bỏ qua ảnh hỏng."
        )
    if errors:
        warnings.warn(f"Bỏ qua {len(errors)}/{n_total} ảnh hỏng. Ví dụ: {errors[0]}")

    valid_mask = np.array([err is None for _, err in results], dtype=bool)
    if not valid_mask.any():
        raise ValueError("Không ảnh nào đọc được — kiểm tra lại đường dẫn dữ liệu.")

    features = np.stack([feat for feat, err in results if err is None]).astype(np.float32)

    if return_valid_mask:
        return features, valid_mask
    return features


# ============================================================
# TEST NHANH — chạy file này trực tiếp để kiểm tra
# python src/features/hog_features.py [duong_dan_anh_1] [duong_dan_anh_2] ...
# Không truyền ảnh → tự lấy tối đa 10 ảnh đầu tiên trong data/raw/train
# ============================================================
if __name__ == "__main__":
    import sys
    import time

    print("=== Test HOG feature extractor ===")

    dims = get_feature_dim()
    print(f"Số chiều lý thuyết (tham số mặc định): {dims}")

    # Lấy ảnh test: từ tham số dòng lệnh, hoặc tự tìm trong dataset
    paths = sys.argv[1:]
    if not paths:
        train_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "train"
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if train_dir.exists():
            paths = [str(p) for p in sorted(train_dir.rglob("*")) if p.suffix.lower() in exts][:10]

    if not paths:
        print("Không tìm thấy ảnh nào để test. Truyền đường dẫn ảnh làm tham số:")
        print("  python src/features/hog_features.py path/to/image.jpg")
        sys.exit(0)

    start = time.time()
    X = batch_extract_features(paths)
    elapsed = time.time() - start

    print(f"Đã xử lý {len(paths)} ảnh trong {elapsed:.2f}s ({elapsed / len(paths) * 1000:.0f} ms/ảnh)")
    print(f"Kích thước 1 vector : {X[0].shape}")
    print(f"Kích thước ma trận  : {X.shape}")
    print(f"Min / Max           : {X.min():.4f} / {X.max():.4f}")
    print(f"Có NaN/Inf không    : {not np.isfinite(X).all()}")
    assert X.shape[1] == dims["total"], "Số chiều thực tế khác số chiều lý thuyết!"
    print("OK — báo cho TN 1 biết số chiều:", X.shape[1])