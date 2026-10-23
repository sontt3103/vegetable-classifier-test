"""
src/evaluation/compare_all.py
-----------------------------
NGƯỜI LÀM: TN 3

So sánh 3 model (Classical SVM, Baseline CNN, MobileNetV2) từ các file report đã có,
KHÔNG cần load lại model, KHÔNG cần TensorFlow.

Chạy:
    python -m src.evaluation.compare_all

Output:
    reports/model_comparison.csv               (đủ 3 model, cùng cột với bản cũ)
    reports/per_class_f1_comparison.csv
    reports/figures/model_comparison_metrics.png
    reports/figures/model_comparison_time.png
    reports/figures/per_class_f1_comparison.png
    reports/evaluation_report.md               (sinh lại, có thêm phần SVM)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import CONFIG

MODELS = [
    ("classical_svm", "Classical SVM"),
    ("baseline_cnn", "Baseline CNN"),
    ("mobilenetv2_transfer", "MobileNetV2"),
]
COLORS = {"Classical SVM": "#E07B39", "Baseline CNN": "#5B8DB8", "MobileNetV2": "#3A9E6F"}


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_table(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for name, display in MODELS:
        s = _load(reports_dir / f"{name}_summary.json")
        m = _load(reports_dir / f"{name}_metadata.json") or {}
        if s is None:
            print(f"[!] Thiếu {name}_summary.json -> bỏ qua {display}")
            continue
        rows.append({
            "Model": name,
            "Display": display,
            "Test Accuracy": s["test_accuracy"],
            "Precision (macro)": s["test_precision_macro"],
            "Recall (macro)": s["test_recall_macro"],
            "F1-score (macro)": s["test_f1_macro"],
            "F1-score (weighted)": s.get("test_f1_weighted"),
            "Training Time (s)": m.get("training_time_seconds"),
            "Num Parameters": m.get("num_parameters"),
            "Test Samples": s.get("num_test_samples"),
            "Misclassified": s.get("num_misclassified"),
            "Inference (ms/img)": m.get("inference_ms_per_image"),
        })
    df = pd.DataFrame(rows)
    if len(df) and df["Test Samples"].nunique() > 1:
        print("[!] CẢNH BÁO: các model đánh giá trên số ảnh test khác nhau:",
              dict(zip(df["Display"], df["Test Samples"])), "-> so sánh KHÔNG công bằng.")
    return df


def plot_metrics(df: pd.DataFrame, save_path: Path):
    metrics = ["Test Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
    x = np.arange(len(metrics))
    w = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - (len(df) - 1) / 2) * w, vals, w, label=row["Display"],
                      color=COLORS.get(row["Display"]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Điểm trên tập test")
    ax.set_title("So sánh 3 model trên cùng tập test")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_time(df: pd.DataFrame, save_path: Path):
    d = df.dropna(subset=["Training Time (s)"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    minutes = d["Training Time (s)"] / 60
    bars = ax.barh(d["Display"], minutes, color=[COLORS.get(n) for n in d["Display"]])
    for b, v in zip(bars, minutes):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.1f} phút", va="center", fontsize=9)
    ax.set_xlabel("Thời gian train (phút)")
    ax.set_title("Thời gian huấn luyện")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, minutes.max() * 1.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def per_class_f1(reports_dir: Path) -> pd.DataFrame:
    cols = {}
    for name, display in MODELS:
        p = reports_dir / f"{name}_per_class_metrics.csv"
        if p.exists():
            cols[display] = pd.read_csv(p).set_index("class")["f1_score"]
    df = pd.DataFrame(cols)
    if not df.empty:
        df = df.sort_values(df.columns[-1])
    return df


def plot_per_class(df: pd.DataFrame, save_path: Path):
    if df.empty:
        return
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(1.6 * len(df.columns) + 3, 0.28 * len(df) + 1.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "F1-score"}, linewidths=0.4)
    ax.set_title("F1-score từng class theo model")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(config: dict = CONFIG):
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(reports_dir)
    if df.empty:
        print("Chưa có report nào.")
        return

    # Giữ đúng các cột report.py đang đọc + thêm cột mới
    df.drop(columns=["Display"]).to_csv(reports_dir / "model_comparison.csv", index=False)
    plot_metrics(df, figures_dir / "model_comparison_metrics.png")
    plot_time(df, figures_dir / "model_comparison_time.png")

    pc = per_class_f1(reports_dir)
    pc.to_csv(reports_dir / "per_class_f1_comparison.csv")
    plot_per_class(pc, figures_dir / "per_class_f1_comparison.png")

    show = df[["Display", "Test Accuracy", "Precision (macro)", "Recall (macro)",
               "F1-score (macro)", "Training Time (s)", "Test Samples"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not pc.empty and len(pc.columns) > 1:
        print("\n5 class khó nhất (F1 trung bình 3 model thấp nhất):")
        print(pc.assign(mean=pc.mean(axis=1)).sort_values("mean").head(5).round(2).to_string())

    try:
        from src.evaluation.report import generate_report
        print(f"\nĐã sinh lại báo cáo: {generate_report(config)}")
    except Exception as e:
        print(f"[!] Không sinh được evaluation_report.md: {e}")


if __name__ == "__main__":
    main()
