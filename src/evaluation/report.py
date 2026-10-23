"""Sinh báo cáo đánh giá dạng Markdown đọc được (reports/evaluation_report.md).

Các file JSON/CSV sinh ra ở bước evaluate rất tốt cho máy đọc, nhưng khó dùng
trực tiếp khi viết báo cáo môn học. Module này gom toàn bộ kết quả lại thành 1
file Markdown có cấu trúc, kèm bảng số liệu và nhúng sẵn các biểu đồ — sinh viên
chỉ cần đọc, bổ sung phần nhận xét của mình, rồi đưa vào báo cáo.

Chạy sau khi đã train + evaluate:
    python -m src.evaluation.report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from configs.config import CONFIG


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(value) -> str:
    return f"{value:.2%}" if isinstance(value, (int, float)) else "—"


def _fmt_num(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame -> bảng Markdown.

    Tự viết thay vì dùng df.to_markdown() để không phụ thuộc package 'tabulate'
    (nếu thiếu package đó, to_markdown sẽ raise và làm hỏng cả báo cáo).
    """
    def fmt_cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = [
        "| " + " | ".join(fmt_cell(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _model_section(name: str, display_name: str, config: dict) -> list[str]:
    reports_dir = Path(config["paths"]["reports_dir"])
    meta = _load_json(reports_dir / f"{name}_metadata.json")
    summary = _load_json(reports_dir / f"{name}_summary.json")

    if meta is None and summary is None:
        return [f"### {display_name}\n", "_Chưa có kết quả — model này chưa được train/đánh giá._\n"]

    lines = [f"### {display_name}\n"]

    if meta:
        lines += [
            "**Cấu hình huấn luyện**\n",
            "| Thông số | Giá trị |",
            "|---|---|",
            f"| Architecture | {meta.get('architecture', '—')} |",
            f"| Số parameters | {meta.get('num_parameters', 0):,} |",
            f"| Input size | {meta.get('input_size', '—')}×{meta.get('input_size', '—')} |",
            f"| Batch size | {meta.get('batch_size', '—')} |",
            f"| Optimizer | {meta.get('optimizer', '—')} |",
            f"| Learning rate | {meta.get('learning_rate') or meta.get('learning_rate_stage1', '—')} |",
            f"| Loss function | {meta.get('loss_function', '—')} |",
            f"| Số epoch đã chạy | {meta.get('epochs_ran') or meta.get('epochs_feature_extraction', '—')} |",
            f"| Thời gian training | {meta.get('training_time_seconds', '—')} giây |",
            f"| Áp dụng class_weight | {'Có' if meta.get('class_weight_applied') else 'Không'} |",
            "",
            "**Kết quả trên tập train/validation**\n",
            "| Metric | Train | Validation |",
            "|---|---:|---:|",
            f"| Accuracy | {_fmt_pct(meta.get('final_train_accuracy'))} | {_fmt_pct(meta.get('final_val_accuracy'))} |",
            f"| Loss | {_fmt_num(meta.get('final_train_loss'))} | {_fmt_num(meta.get('final_val_loss'))} |",
            "",
            f"![Training curves](figures/{name}_training_curves.png)\n",
        ]

    if summary:
        lines += [
            "**Kết quả trên tập test độc lập**\n",
            "| Metric | Giá trị |",
            "|---|---:|",
            f"| Accuracy | {_fmt_pct(summary.get('test_accuracy'))} |",
            f"| Precision (macro) | {_fmt_num(summary.get('test_precision_macro'))} |",
            f"| Recall (macro) | {_fmt_num(summary.get('test_recall_macro'))} |",
            f"| F1-score (macro) | {_fmt_num(summary.get('test_f1_macro'))} |",
            f"| F1-score (weighted) | {_fmt_num(summary.get('test_f1_weighted'))} |",
            f"| Số ảnh test | {summary.get('num_test_samples', '—')} |",
            f"| Số ảnh dự đoán sai | {summary.get('num_misclassified', '—')} |",
            "",
            f"- **Class có kết quả tốt nhất:** {summary.get('best_class', '—')}",
            f"- **Class yếu nhất:** {summary.get('worst_class', '—')}",
            "",
        ]

        overfit = summary.get("overfitting_analysis")
        if overfit:
            status_label = {
                "overfitting": "⚠️ Overfitting",
                "underfitting": "⚠️ Underfitting",
                "good_fit": "✅ Tổng quát hoá tốt",
            }.get(overfit.get("status"), overfit.get("status"))
            lines += [
                f"**Phân tích overfitting/underfitting:** {status_label}\n",
                f"> {overfit.get('explanation', '')}\n",
            ]

        confused = summary.get("top_confused_pairs") or []
        if confused:
            lines += [
                "**Các cặp class bị nhầm lẫn nhiều nhất**\n",
                "| Nhãn thật | Model dự đoán thành | Số lần |",
                "|---|---|---:|",
            ]
            lines += [
                f"| {p['true_class']} | {p['predicted_class']} | {p['count']} |"
                for p in confused
            ]
            lines.append("")

        lines += [
            f"![Confusion matrix](figures/{name}_confusion_matrix.png)\n",
            f"![Dự đoán sai](figures/{name}_misclassified.png)\n",
        ]

    return lines


def generate_report(config: dict = CONFIG) -> Path:
    """Tổng hợp toàn bộ kết quả thành reports/evaluation_report.md."""
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("baseline_cnn", "Model 1 — CNN Baseline (tự xây)"),
        ("mobilenetv2_transfer", "Model 2 — MobileNetV2 (Transfer Learning)"),
    ]

    lines = [
        "# Báo cáo Đánh giá Model — Phân loại Rau Củ Quả\n",
        "> File này được sinh tự động bởi `python -m src.evaluation.report`.",
        "> Các phần _Nhận xét_ ở cuối cần bạn tự viết dựa trên số liệu bên trên.\n",
        "---\n",
        "## 1. Dữ liệu đánh giá\n",
    ]

    # Thông tin dataset lấy từ metadata của model bất kỳ đã train
    for name, _ in models:
        meta = _load_json(reports_dir / f"{name}_metadata.json")
        if meta and meta.get("class_names"):
            lines += [
                f"- Số class: **{len(meta['class_names'])}**",
                f"- Danh sách class: {', '.join(meta['class_names'])}",
                "",
                "![Phân bố class](figures/class_distribution_train.png)\n",
            ]
            break
    else:
        lines.append("_Chưa có model nào được train — chưa xác định được thông tin dataset._\n")

    lines += ["---\n", "## 2. Kết quả từng model\n"]
    for name, display in models:
        lines += _model_section(name, display, config)
        lines.append("---\n")

    # ---------------- Bảng so sánh ----------------
    lines.append("## 3. So sánh model\n")
    comparison_path = reports_dir / "model_comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        lines += [_df_to_markdown(df), ""]

        if len(df) >= 2:
            best_acc = df.loc[df["Test Accuracy"].idxmax()]
            best_f1 = df.loc[df["F1-score (macro)"].idxmax()]
            fastest = df.loc[df["Training Time (s)"].idxmin()] if df["Training Time (s)"].notna().all() else None

            lines += [
                "\n**Quan sát tự động từ số liệu:**\n",
                f"- Accuracy cao nhất: **{best_acc['Model']}** ({best_acc['Test Accuracy']:.2%})",
                f"- F1-score (macro) cao nhất: **{best_f1['Model']}** ({best_f1['F1-score (macro)']:.4f})",
            ]
            if fastest is not None:
                lines.append(f"- Train nhanh nhất: **{fastest['Model']}** ({fastest['Training Time (s)']:.0f} giây)")
            lines += [
                "",
                "> **Lưu ý khi chọn final model:** accuracy cao nhất KHÔNG đương nhiên là model tốt nhất.",
                "> Nếu dataset mất cân bằng, F1-score (macro) phản ánh chất lượng công bằng hơn giữa các class.",
                "> Cũng cần cân nhắc thời gian training, kích thước model, và tốc độ inference khi deploy.\n",
            ]
    else:
        lines.append("_Chưa có bảng so sánh — chạy `python -m src.evaluation.evaluate --compare` trước._\n")

    # ---------------- Phần sinh viên tự viết ----------------
    lines += [
        "---\n",
        "## 4. Nhận xét & Kết luận _(bạn tự viết phần này)_\n",
        "**4.1. Model nào tốt nhất và vì sao?**\n",
        "_(dựa vào bảng so sánh mục 3, giải thích lựa chọn final model)_\n",
        "**4.2. Vì sao một số class dễ bị nhầm lẫn?**\n",
        "_(xem bảng cặp class nhầm lẫn + ảnh dự đoán sai ở mục 2 — ví dụ: hình dạng/màu sắc tương tự nhau)_\n",
        "**4.3. Model có overfitting/underfitting không?**\n",
        "_(xem phân tích tự động ở mục 2 và biểu đồ training curves)_\n",
        "**4.4. Hạn chế của model**\n",
        "_(ví dụ: dataset nhỏ, ảnh nền đơn giản không giống thực tế, một số class rất giống nhau...)_\n",
        "**4.5. Hướng phát triển tiếp theo**\n",
        "_(ví dụ: thu thập thêm ảnh thực tế, augmentation mạnh hơn, thử EfficientNet, deploy lên mobile bằng TensorFlow Lite...)_\n",
    ]

    out_path = reports_dir / "evaluation_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = generate_report()
    print(f"✓ Đã sinh báo cáo: {path}")
