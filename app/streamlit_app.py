"""Streamlit demo application — Hệ thống phân loại rau củ quả.

Chạy:
    streamlit run app/streamlit_app.py

Ứng dụng được thiết kế như một application hoàn chỉnh (không phải giao diện thử
nghiệm sơ sài), gồm 3 tab:

    1. Dự đoán       — upload ảnh, preview, predict, confidence, Top-K
    2. Thông tin Model — architecture, parameters, hyperparameters, danh sách class
    3. Kết quả đánh giá — metrics trên test set, confusion matrix, training curves

App luôn dùng ĐÚNG final model đã được đánh giá trong `src/evaluation` — số liệu
hiển thị ở tab 3 được đọc trực tiếp từ `reports/`, nên không thể lệch với model
đang chạy inference ở tab 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Cho phép chạy `streamlit run app/streamlit_app.py` từ bất kỳ đâu
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG  # noqa: E402
from src.inference.predict import VegetableClassifier, InvalidImageError  # noqa: E402
from src.inference.svm_predict import SVMVegetableClassifier, find_svm_checkpoint  # noqa: E402

st.set_page_config(
    page_title="Phân loại Rau Củ Quả",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Bảng màu — dùng chung cho biểu đồ Top-K và các thành phần UI.
# Đây là bảng màu đã kiểm định đảm bảo người bị mù màu vẫn phân biệt được,
# và đủ tương phản trên cả nền sáng lẫn nền tối.
# ---------------------------------------------------------------------------
ACCENT = "#2a78d6"          # màu nhấn — dành cho kết quả dự đoán đứng đầu
MUTED_BAR = "#c3c2b7"       # màu xám — các kết quả còn lại (đóng vai trò ngữ cảnh)
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

CUSTOM_CSS = """
<style>
  .app-header {
      padding: 1.5rem 1.75rem;
      border-radius: 14px;
      background: linear-gradient(135deg, #1baf7a 0%, #2a78d6 100%);
      color: #ffffff;
      margin-bottom: 1.25rem;
  }
  .app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
  .app-header p  { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

  .result-card {
      border: 1px solid rgba(11,11,11,0.10);
      border-radius: 14px;
      padding: 1.25rem 1.5rem;
      background: #fcfcfb;
  }
  .result-label {
      font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: #52514e; margin-bottom: 0.25rem;
  }
  .result-class {
      font-size: 2rem; font-weight: 700; color: #0b0b0b;
      line-height: 1.2; margin-bottom: 0.25rem;
  }
  .result-confidence { font-size: 1.1rem; font-weight: 600; }

  /* Biểu đồ thanh ngang cho Top-K: nhãn trực tiếp trên từng thanh,
     thanh dẫn đầu dùng màu nhấn, các thanh còn lại xám (dạng "emphasis"). */
  .bar-row { margin-bottom: 0.7rem; }
  .bar-head {
      display: flex; justify-content: space-between;
      font-size: 0.9rem; margin-bottom: 0.25rem; color: #0b0b0b;
  }
  .bar-head .pct { font-variant-numeric: tabular-nums; color: #52514e; }
  .bar-track {
      height: 12px; background: #f0efec; border-radius: 6px; overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 6px; }

  .empty-state {
      border: 2px dashed rgba(11,11,11,0.15);
      border-radius: 14px; padding: 2.5rem 1.5rem; text-align: center;
      color: #52514e;
  }
</style>
"""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Thứ tự ưu tiên khi chọn model cho demo. Sau khi so sánh 2 model xong
# (reports/model_comparison.csv), đặt FINAL MODEL bạn đã chọn lên đầu danh sách.
CHECKPOINT_PRIORITY = ["mobilenetv2_transfer", "baseline_cnn", "svm"]

# Tên model -> file checkpoint (SVM là .pkl và cần thêm pca_model.pkl)
CHECKPOINT_FILES = {
    "mobilenetv2_transfer": "mobilenetv2_transfer.keras",
    "baseline_cnn": "baseline_cnn.keras",
    "svm": "svm_model.pkl",
}
DISPLAY_NAMES = {
    "mobilenetv2_transfer": "MobileNetV2 (Transfer Learning)",
    "baseline_cnn": "CNN Baseline",
    "svm": "HOG → PCA → SVM (ML cổ điển)",
}

# Ngưỡng (thấp, cao) để cảnh báo độ tin cậy. SVM dùng xác suất Platt scaling trên
# 36 class nên xác suất thường thấp hơn softmax của CNN -> dùng ngưỡng thấp hơn.
# Đây là giá trị tạm: chỉnh lại sau khi xem kết quả thật của SVM.
CONF_THRESHOLDS = {"svm": (0.2, 0.4), "default": (0.4, 0.7)}


def conf_thresholds(model_name: str) -> tuple[float, float]:
    return CONF_THRESHOLDS.get(model_name, CONF_THRESHOLDS["default"])


@st.cache_resource(show_spinner="Đang load model...")
def get_classifier(checkpoint_path: str):
    if str(checkpoint_path).endswith(".pkl"):
        return SVMVegetableClassifier(checkpoint_path)
    return VegetableClassifier(checkpoint_path)


@st.cache_data
def load_report(name: str, suffix: str) -> dict | None:
    path = Path(CONFIG["paths"]["reports_dir"]) / f"{name}_{suffix}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_available_checkpoints() -> list[tuple[str, Path]]:
    """Trả về [(tên_model, đường_dẫn)] các checkpoint hiện có, theo thứ tự ưu tiên."""
    ckpt_dir = Path(CONFIG["paths"]["checkpoints_dir"])
    found = []
    for name in CHECKPOINT_PRIORITY:
        if name == "svm":
            # classical_svm.pkl (train_svm.py) hoặc cặp svm_model.pkl + pca_model.pkl
            path = find_svm_checkpoint(ckpt_dir)
        else:
            path = ckpt_dir / CHECKPOINT_FILES[name]
            path = path if path.exists() else None
        if path is not None:
            found.append((name, path))
    return found


def normalize_svm_summary(raw: dict) -> dict:
    """Đưa svm_evaluation.json (TN 3) về cùng định dạng *_summary.json của CNN.

    Chưa biết chính xác tên key TN 3 sẽ sinh ra, nên thử nhiều tên phổ biến.
    Sau khi có file thật, chỉnh danh sách key bên dưới cho khớp.
    """
    def pick(*keys):
        for k in keys:
            if k in raw and raw[k] is not None:
                return raw[k]
        return None

    # Nếu có classification_report (classical_svm_evaluation.json của TN 3) thì suy ra
    # số ảnh test, số ảnh sai, class tốt nhất/yếu nhất
    report = raw.get("classification_report") or {}
    per_class = {
        k: v for k, v in report.items()
        if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
    }
    n_test = None
    if isinstance(report.get("macro avg"), dict):
        n_test = int(report["macro avg"].get("support") or 0) or None
    best = worst = None
    if per_class:
        best = max(per_class, key=lambda k: per_class[k].get("f1-score", 0))
        worst = min(per_class, key=lambda k: per_class[k].get("f1-score", 0))
    acc = pick("test_accuracy", "accuracy")
    n_wrong = None
    if n_test and isinstance(acc, (int, float)):
        n_wrong = int(round(n_test * (1 - acc)))

    def first(*vals):
        return next((v for v in vals if v is not None), None)

    summary = {
        "test_accuracy": acc,
        "test_precision_macro": pick("test_precision_macro", "precision_macro", "precision"),
        "test_recall_macro": pick("test_recall_macro", "recall_macro", "recall"),
        "test_f1_macro": pick("test_f1_macro", "f1_macro", "f1_score", "f1"),
        "num_test_samples": first(pick("num_test_samples", "n_test"), n_test),
        "num_misclassified": first(pick("num_misclassified"), n_wrong),
        "best_class": first(pick("best_class"), best),
        "worst_class": first(pick("worst_class"), worst),
        "top_confused_pairs": pick("top_confused_pairs"),
        "overfitting_analysis": pick("overfitting_analysis"),
        # Thông tin riêng của SVM trong svm_evaluation.json (classical_svm.py)
        "svm_extra": {
            "train_time_minutes": pick("train_time_minutes"),
            "best_cv_f1": pick("best_cv_f1"),
            "pca_variance_kept": pick("pca_variance_kept"),
        },
    }
    if summary["test_accuracy"] is None:
        return {}
    # Bỏ các key không có giá trị để phần hiển thị dùng được .get(key, "—")
    return {k: v for k, v in summary.items() if v is not None}


def render_topk_chart(top_k: list[dict]):
    """Biểu đồ thanh ngang cho Top-K predictions.

    Dùng dạng "emphasis": kết quả đứng đầu là thứ người dùng quan tâm nên được
    tô màu nhấn, các kết quả còn lại giữ màu xám làm ngữ cảnh so sánh. Mỗi thanh
    đều có nhãn trực tiếp (tên class + %), nên thông tin không phụ thuộc vào màu.
    """
    html = []
    for i, item in enumerate(top_k):
        pct = item["probability"] * 100
        color = ACCENT if i == 0 else MUTED_BAR
        html.append(
            f'<div class="bar-row">'
            f'  <div class="bar-head"><span>{item["class"]}</span>'
            f'  <span class="pct">{pct:.2f}%</span></div>'
            f'  <div class="bar-track">'
            f'    <div class="bar-fill" style="width:{max(pct, 0.6):.2f}%;background:{color};"></div>'
            f'  </div>'
            f'</div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def confidence_color(confidence: float, model_name: str = "default") -> str:
    low, high = conf_thresholds(model_name)
    if confidence >= high:
        return GOOD
    if confidence >= low:
        return WARNING
    return CRITICAL


# ---------------------------------------------------------------------------
# Tab 1 — Dự đoán
# ---------------------------------------------------------------------------
def tab_predict(classifier, top_k: int, model_name: str = "default"):
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Chọn ảnh")
        uploaded_file = st.file_uploader(
            "Upload ảnh rau/củ/quả",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Hỗ trợ JPG, PNG, BMP, WEBP. Ảnh nên rõ nét và chứa 1 loại rau/quả chính.",
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Preview — {uploaded_file.name}", use_container_width=True)
                st.caption(f"Kích thước gốc: {image.size[0]}×{image.size[1]} px · Định dạng: {image.format}")
            except Exception:  # noqa: BLE001
                st.error(
                    "❌ Không thể mở ảnh này. File có thể bị hỏng hoặc không phải "
                    "định dạng ảnh hợp lệ. Hãy thử một ảnh khác."
                )
                image = None
        else:
            st.markdown(
                '<div class="empty-state">📁<br><br>Chưa có ảnh nào được chọn.<br>'
                'Hãy upload một ảnh để bắt đầu.</div>',
                unsafe_allow_html=True,
            )

        predict_clicked = st.button(
            "🔍 Dự đoán", type="primary", use_container_width=True, disabled=image is None
        )

    with right:
        st.subheader("2. Kết quả")

        if not predict_clicked:
            st.markdown(
                '<div class="empty-state">Kết quả dự đoán sẽ hiển thị ở đây<br>'
                'sau khi bạn bấm nút <b>Dự đoán</b>.</div>',
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Đang phân tích ảnh..."):
            try:
                result = classifier.predict(image, top_k=top_k)
            except InvalidImageError as e:
                st.error(f"❌ Ảnh không hợp lệ: {e}")
                return
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ Có lỗi xảy ra khi dự đoán: {e}")
                return

        color = confidence_color(result.confidence, model_name)
        low_thr, high_thr = conf_thresholds(model_name)
        st.markdown(
            f'<div class="result-card">'
            f'  <div class="result-label">Kết quả dự đoán</div>'
            f'  <div class="result-class">{result.predicted_class}</div>'
            f'  <div class="result-confidence" style="color:{color};">'
            f'    Độ tin cậy: {result.confidence:.2%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if result.confidence < low_thr:
            st.warning(
                "⚠️ **Độ tin cậy thấp.** Ảnh có thể không thuộc các loại rau/quả model đã "
                "được học, hoặc chất lượng ảnh chưa tốt (mờ, thiếu sáng, có nhiều vật thể). "
                "Kết quả này không đáng tin cậy."
            )
        elif result.confidence < high_thr:
            st.info(
                "ℹ️ Độ tin cậy ở mức trung bình — nên tham khảo thêm các kết quả Top-K bên dưới."
            )

        st.markdown(f"##### Top-{len(result.top_k)} kết quả có khả năng nhất")
        render_topk_chart(result.top_k)

        # Lưu vào lịch sử phiên làm việc
        st.session_state.setdefault("history", [])
        st.session_state["history"].insert(0, {
            "model": DISPLAY_NAMES.get(model_name, model_name),
            "class": result.predicted_class,
            "confidence": result.confidence,
        })
        st.session_state["history"] = st.session_state["history"][:10]

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.markdown("##### 🕘 Lịch sử dự đoán trong phiên này")
        st.dataframe(
            [
                {
                    "#": i + 1,
                    "Model": h.get("model", "—"),
                    "Loại rau/quả": h["class"],
                    "Độ tin cậy": f"{h['confidence']:.2%}",
                }
                for i, h in enumerate(history)
            ],
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Thông tin Model
# ---------------------------------------------------------------------------
def tab_model_info(classifier: VegetableClassifier, model_name: str):
    info = classifier.model_info()
    meta = load_report(model_name, "metadata")

    st.subheader("Thông tin model đang sử dụng")

    c1, c2, c3 = st.columns(3)
    c1.metric("Số class", info["num_classes"])
    c2.metric("Input size", f"{info['input_size']}×{info['input_size']}")
    if info.get("kind") == "svm":
        c3.metric("Support vectors", f"{info['num_support_vectors']:,}" if info.get("num_support_vectors") else "—")
        st.markdown("##### Cấu hình pipeline HOG → PCA → SVM")
        params = info.get("best_params") or {}
        rows = [
            ("Đặc trưng", "HOG + Color Histogram"),
            ("Số chiều sau PCA", info.get("pca_components") or "—"),
            ("Kernel", info.get("kernel") or "—"),
            ("C tốt nhất (GridSearchCV)", params.get("C", "—")),
            ("Gamma tốt nhất (GridSearchCV)", params.get("gamma", "—")),
        ]
        if "color_w" in params:
            rows.append(("Trọng số màu (Hellinger)", params["color_w"]))
        if info.get("architecture"):
            rows.append(("Kiến trúc", info["architecture"]))
        st.dataframe(
            [{"Thông số": k, "Giá trị": str(v)} for k, v in rows],
            hide_index=True, use_container_width=True,
        )
    else:
        c3.metric("Số parameters", f"{info['num_parameters']:,}")

    if meta and info.get("kind") != "svm":
        st.markdown("##### Cấu hình huấn luyện")
        rows = [
            ("Architecture", meta.get("architecture", "—")),
            ("Optimizer", meta.get("optimizer", "—")),
            ("Learning rate", meta.get("learning_rate") or meta.get("learning_rate_stage1", "—")),
            ("Loss function", meta.get("loss_function", "—")),
            ("Batch size", meta.get("batch_size", "—")),
            ("Số epoch đã chạy", meta.get("epochs_ran") or meta.get("epochs_feature_extraction", "—")),
            ("Thời gian training", f"{meta.get('training_time_seconds', '—')} giây"),
            ("Áp dụng class_weight", "Có" if meta.get("class_weight_applied") else "Không"),
        ]
        st.dataframe(
            [{"Thông số": k, "Giá trị": str(v)} for k, v in rows],
            hide_index=True, use_container_width=True,
        )

        st.markdown("##### Kết quả train / validation")
        m1, m2 = st.columns(2)
        if isinstance(meta.get("final_train_accuracy"), (int, float)):
            m1.metric("Train accuracy", f"{meta['final_train_accuracy']:.2%}")
        if isinstance(meta.get("final_val_accuracy"), (int, float)):
            m2.metric("Validation accuracy", f"{meta['final_val_accuracy']:.2%}")
    elif info.get("kind") != "svm":
        st.info(
            "Chưa tìm thấy file metadata của model này trong `reports/`. "
            "Train lại model để sinh đầy đủ thông tin huấn luyện."
        )

    st.markdown("##### Danh sách các class model có thể nhận diện")
    st.dataframe(
        [{"#": i + 1, "Tên class": c} for i, c in enumerate(info["class_names"])],
        hide_index=True, use_container_width=True, height=280,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Kết quả đánh giá
# ---------------------------------------------------------------------------
def tab_evaluation(model_name: str):
    st.subheader("Kết quả đánh giá trên tập test độc lập")

    figures_dir = Path(CONFIG["paths"]["figures_dir"])

    if model_name == "svm":
        # Kết quả SVM có thể nằm ở một trong các file sau (thử theo thứ tự):
        #   classical_svm_summary.json  (do scripts/convert_svm_report.py sinh, cùng dạng CNN)
        #   svm_evaluation.json         (classical_svm.py của TN 1)
        #   classical_svm_evaluation.json (TN 3)
        raw = None
        for fname in ("classical_svm_summary.json", "svm_evaluation.json", "classical_svm_evaluation.json"):
            path = Path(CONFIG["paths"]["reports_dir"]) / fname
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                break
        summary = normalize_svm_summary(raw) if raw else None
    else:
        summary = load_report(model_name, "summary")

    if not summary:
        st.info(
            "Chưa có kết quả đánh giá cho model này.\n\n"
            + (
                "Kết quả SVM được lưu ở `reports/svm_evaluation.json` sau khi train xong "
                "(`python -m src.models.classical_svm`)."
                if model_name == "svm"
                else "Chạy lệnh sau để sinh đầy đủ metrics và biểu đồ:\n\n"
                "```bash\npython -m src.evaluation.evaluate --compare\n```"
            )
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{summary['test_accuracy']:.2%}")
        def _fmt(key):
            v = summary.get(key)
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

        c2.metric("Precision (macro)", _fmt("test_precision_macro"))
        c3.metric("Recall (macro)", _fmt("test_recall_macro"))
        c4.metric("F1-score (macro)", _fmt("test_f1_macro"))

        # SVM chưa có các trường này trong svm_evaluation.json -> ẩn dòng thay vì hiện "—"
        if "num_test_samples" in summary:
            st.caption(
                f"Đánh giá trên {summary.get('num_test_samples', '—')} ảnh test · "
                f"{summary.get('num_misclassified', '—')} ảnh bị dự đoán sai · "
                f"Class tốt nhất: **{summary.get('best_class', '—')}** · "
                f"Class yếu nhất: **{summary.get('worst_class', '—')}**"
            )

        extra = summary.get("svm_extra") or {}
        parts = []
        if isinstance(extra.get("train_time_minutes"), (int, float)):
            parts.append(f"Thời gian train: **{extra['train_time_minutes']:.1f} phút**")
        if isinstance(extra.get("best_cv_f1"), (int, float)):
            parts.append(f"F1 cross-validation: **{extra['best_cv_f1']:.3f}**")
        if isinstance(extra.get("pca_variance_kept"), (int, float)):
            parts.append(f"Phương sai PCA giữ lại: **{extra['pca_variance_kept']:.1%}**")
        if parts:
            st.caption(" · ".join(parts))

        overfit = summary.get("overfitting_analysis")
        if overfit:
            status = overfit.get("status")
            message = overfit.get("explanation", "")
            if status == "good_fit":
                st.success(f"✅ {message}")
            else:
                st.warning(f"⚠️ {message}")

        confused = summary.get("top_confused_pairs") or []
        if confused:
            st.markdown("##### Các cặp class bị nhầm lẫn nhiều nhất")
            st.dataframe(
                [
                    {
                        "Nhãn thật": p["true_class"],
                        "Model dự đoán thành": p["predicted_class"],
                        "Số lần": p["count"],
                    }
                    for p in confused
                ],
                hide_index=True, use_container_width=True,
            )

    st.divider()
    st.markdown("##### Biểu đồ")

    # SVM có thể đặt tên file hình theo "svm_" hoặc "classical_svm_"
    prefixes = ["svm", "classical_svm"] if model_name == "svm" else [model_name]
    figures = [
        (f"{prefix}_{suffix}.png", caption)
        for prefix in prefixes
        for suffix, caption in [
            ("training_curves", "Training vs Validation (Accuracy & Loss)"),
            ("confusion_matrix", "Confusion Matrix"),
            ("misclassified", "Các ảnh bị dự đoán sai"),
            ("correct_predictions", "Các ảnh dự đoán đúng"),
        ]
    ]

    any_figure = False
    for filename, caption in figures:
        path = figures_dir / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
            any_figure = True

    if not any_figure:
        if model_name == "svm":
            st.info(
                "Chưa có biểu đồ cho SVM. Đặt ảnh vào `reports/figures/` với tên "
                "`svm_confusion_matrix.png` hoặc `classical_svm_confusion_matrix.png` "
                "(và `..._misclassified.png` nếu có) để hiển thị ở đây."
            )
        else:
            st.info(
                "Chưa có biểu đồ nào. Chạy `python -m src.training.train --model both` "
                "rồi `python -m src.evaluation.evaluate --compare` để sinh biểu đồ."
            )


# ---------------------------------------------------------------------------
# Tab 4 — So sánh các model trên cùng 1 ảnh
# ---------------------------------------------------------------------------
def tab_compare(available: list[tuple[str, Path]], top_k: int):
    st.subheader("So sánh các model trên cùng một ảnh")
    st.caption("Upload 1 ảnh, mỗi model sẽ tự dự đoán — dùng để chụp màn hình cho báo cáo/PPT.")

    uploaded = st.file_uploader(
        "Upload ảnh rau/củ/quả",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="compare_uploader",
    )
    if uploaded is None:
        st.markdown(
            '<div class="empty-state">📁<br><br>Hãy upload một ảnh để so sánh các model.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        image = Image.open(uploaded)
        image.load()
    except Exception:  # noqa: BLE001
        st.error("❌ Không thể mở ảnh này. Hãy thử một ảnh khác.")
        return

    st.image(image, caption=uploaded.name, width=280)

    cols = st.columns(len(available))
    top1 = []
    for col, (name, path) in zip(cols, available):
        with col:
            st.markdown(f"##### {DISPLAY_NAMES.get(name, name)}")
            try:
                result = get_classifier(str(path)).predict(image, top_k=top_k)
            except InvalidImageError as e:
                st.error(f"❌ Ảnh không hợp lệ: {e}")
                continue
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ Lỗi khi dự đoán: {e}")
                continue
            top1.append(result.predicted_class)
            color = confidence_color(result.confidence, name)
            st.markdown(
                f'<div class="result-card">'
                f'  <div class="result-class" style="font-size:1.5rem;">{result.predicted_class}</div>'
                f'  <div class="result-confidence" style="color:{color};">'
                f'    Độ tin cậy: {result.confidence:.2%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            render_topk_chart(result.top_k)

    if len(top1) > 1:
        if len(set(top1)) == 1:
            st.success(f"✅ Các model đều dự đoán: **{top1[0]}**")
        else:
            st.warning("⚠️ Các model cho kết quả khác nhau — đây là ca hay để phân tích trong báo cáo.")


# ---------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="app-header">'
        '<h1>🥕 Hệ thống Phân loại Rau Củ Quả</h1>'
        '<p>Đồ án học phần Học máy — Nhận diện rau/củ/quả từ ảnh: SVM, CNN và Transfer Learning</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    available = find_available_checkpoints()

    if not available:
        st.error("⚠️ Chưa tìm thấy model đã train nào trong thư mục `checkpoints/`.")
        st.markdown(
            "**Cách khắc phục** — chạy lần lượt các lệnh sau ở thư mục gốc project:\n\n"
            "```bash\n"
            "python scripts/download_dataset.py       # tải dataset\n"
            "python -m src.training.train --model both  # train model\n"
            "```\n\n"
            "Xem hướng dẫn chi tiết trong `TRAINING_GUIDE.md`."
        )
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Cấu hình")

        options = [name for name, _ in available]
        selected = st.selectbox(
            "Model sử dụng",
            options,
            format_func=lambda n: DISPLAY_NAMES.get(n, n),
            help="Chọn model để dự đoán. Tab *So sánh model* chạy tất cả model cùng lúc.",
        )
        if not any(n == "svm" for n in options):
            st.caption(
                "ℹ️ Chưa thấy model SVM (cần `checkpoints/classical_svm.pkl`, "
                "hoặc cả `svm_model.pkl` + `pca_model.pkl`)."
            )
        checkpoint_path = dict(available)[selected]

        classifier = None
        try:
            classifier = get_classifier(str(checkpoint_path))
        except Exception as e:  # noqa: BLE001
            st.error(f"Không load được model: {e}")

        if classifier is not None:
            max_k = min(10, classifier.model_info()["num_classes"])
            top_k = st.slider("Số kết quả Top-K", 1, max_k, min(3, max_k))
        else:
            top_k = 5

        st.divider()
        st.caption(
            "Ứng dụng sử dụng đúng model đã được đánh giá trong `src/evaluation` — "
            "số liệu ở tab *Kết quả đánh giá* được đọc trực tiếp từ `reports/`."
        )

    if classifier is None:
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Dự đoán", "🧠 Thông tin Model", "📊 Kết quả đánh giá", "⚖️ So sánh model"]
    )
    with tab1:
        tab_predict(classifier, top_k, selected)
    with tab2:
        tab_model_info(classifier, selected)
    with tab3:
        tab_evaluation(selected)
    with tab4:
        tab_compare(available, top_k)


if __name__ == "__main__":
    main()
