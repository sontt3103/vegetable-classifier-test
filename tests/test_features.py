"""Test cho src/features/ (HOG + Color Histogram + PCAWrapper) — phần của TN 2.

Chạy: pytest tests/test_features.py
Tự sinh ảnh giả trong thư mục tạm nên không cần dataset thật.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.features.hog_features import (
    COLOR_WEIGHT,
    DEFAULT_IMG_SIZE,
    batch_extract_features,
    extract_color_histogram,
    extract_combined_features,
    extract_hog_features,
    get_feature_dim,
    load_and_resize,
)
from src.features.pca_wrapper import PCAWrapper


@pytest.fixture
def image_files(tmp_path):
    """4 ảnh kích thước khác nhau (kiểm tra resize) + 1 ảnh grayscale."""
    rng = np.random.default_rng(0)
    paths = []
    for i, (w, h) in enumerate([(200, 300), (100, 100), (64, 48)]):
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    gray = tmp_path / "gray.png"
    Image.fromarray(rng.integers(0, 255, (80, 80), dtype=np.uint8)).save(gray)
    paths.append(str(gray))
    return paths


def test_load_and_resize_shape_and_dtype(image_files):
    for p in image_files:
        img = load_and_resize(p)
        assert img.shape == (DEFAULT_IMG_SIZE[1], DEFAULT_IMG_SIZE[0], 3)
        assert img.dtype == np.uint8


def test_load_and_resize_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_resize(str(tmp_path / "khong_ton_tai.jpg"))
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"day khong phai anh")
    with pytest.raises(ValueError):
        load_and_resize(str(bad))


def test_feature_dimensions_match_documentation(image_files):
    dims = get_feature_dim()
    assert dims == {"hog": 8100, "color": 96, "total": 8196}
    vec = extract_combined_features(image_files[0])
    assert vec.shape == (dims["total"],)
    assert np.isfinite(vec).all()


def test_color_histogram_is_proportion_per_channel():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0] = 255                       # toàn đỏ
    hist = extract_color_histogram(img, bins=32).reshape(3, 32)
    assert np.allclose(hist.sum(axis=1), 1.0)
    assert hist[0, -1] == pytest.approx(1.0)  # kênh R dồn vào bin cuối
    assert hist[1, 0] == pytest.approx(1.0)   # kênh G dồn vào bin đầu


def test_color_weight_scales_only_color_part(image_files):
    plain = extract_combined_features(image_files[0], color_weight=1.0)
    weighted = extract_combined_features(image_files[0], color_weight=COLOR_WEIGHT)
    np.testing.assert_allclose(weighted[:8100], plain[:8100])                 # HOG không đổi
    np.testing.assert_allclose(weighted[8100:], plain[8100:] * COLOR_WEIGHT, rtol=1e-5)
    assert weighted.shape == plain.shape == (8196,)                            # số chiều không đổi


def test_hog_visualize_returns_image(image_files):
    img = load_and_resize(image_files[0])
    vec, hog_img = extract_hog_features(img, visualize=True)
    assert vec.shape == (8100,)
    assert hog_img.shape == img.shape[:2]


def test_features_are_deterministic(image_files):
    a = extract_combined_features(image_files[0])
    b = extract_combined_features(image_files[0])
    np.testing.assert_array_equal(a, b)


def test_batch_matches_single_and_skips_bad(image_files, tmp_path):
    X = batch_extract_features(image_files, show_progress=False)
    assert X.shape == (len(image_files), 8196)
    np.testing.assert_allclose(X[0], extract_combined_features(image_files[0]))

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        batch_extract_features(image_files + [str(bad)], show_progress=False)

    with pytest.warns(UserWarning):
        X2, mask = batch_extract_features(
            image_files + [str(bad)], show_progress=False,
            on_error="skip", return_valid_mask=True,
        )
    assert X2.shape[0] == len(image_files)
    assert mask.tolist() == [True] * len(image_files) + [False]


def test_pca_shapes_and_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 300))
    pca = PCAWrapper(n_components=20)
    Z = pca.fit_transform(X)
    assert Z.shape == (60, 20)
    assert 0 < pca.get_total_variance_kept() <= 1

    path = tmp_path / "sub" / "pca.pkl"
    pca.save(str(path))
    loaded = PCAWrapper.load(str(path))
    np.testing.assert_allclose(loaded.transform(X), Z)


def test_pca_variance_mode_and_none():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 10)) @ rng.normal(size=(10, 100))
    for nc in (0.95, None):
        pca = PCAWrapper(n_components=nc).fit(X)
        assert pca.get_total_variance_kept() >= 0.95
        assert pca.n_components_ <= 10


def test_pca_clamps_too_many_components():
    X = np.random.default_rng(0).normal(size=(10, 50))
    with pytest.warns(UserWarning):
        Z = PCAWrapper(n_components=200).fit_transform(X)
    assert Z.shape == (10, 10)


def test_pca_guards():
    X = np.random.default_rng(0).normal(size=(20, 30))
    with pytest.raises(RuntimeError):
        PCAWrapper(5).transform(X)
    pca = PCAWrapper(5).fit(X)
    with pytest.raises(ValueError):
        pca.transform(np.zeros((2, 7)))      # sai số chiều
    with pytest.raises(ValueError):
        pca.transform(np.full((2, 30), np.nan))