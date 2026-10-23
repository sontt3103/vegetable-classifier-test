"""Vẽ biểu đồ Training vs Validation (Accuracy & Loss).

Đây là yêu cầu bắt buộc của môn học (mục 7 — Visualization / Training):
    - Training vs Validation Accuracy
    - Training vs Validation Loss
và dùng để trả lời câu hỏi "model có bị overfitting/underfitting không?".

Nguyên tắc thiết kế biểu đồ áp dụng ở đây:
    - KHÔNG dùng 2 trục y trên cùng 1 biểu đồ (accuracy và loss khác đơn vị
      hoàn toàn) -> tách thành 2 biểu đồ riêng.
    - 2 đường (train / validation) dùng 2 màu phân biệt rõ, đã kiểm tra
      đảm bảo người bị mù màu vẫn phân biệt được.
    - Luôn có legend, title, nhãn trục.
    - Grid mờ, không lấn át dữ liệu.
    - Đánh dấu trực tiếp epoch tốt nhất thay vì ghi số lên mọi điểm.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from configs.config import CONFIG

# Bảng màu đã được kiểm định (CVD-safe): xanh dương = train, cam = validation.
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"
COLOR_SURFACE = "#fcfcfb"


def _style_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=12, color=COLOR_TEXT, pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=COLOR_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=COLOR_MUTED)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    # Epoch là số nguyên — không hiển thị tick lẻ kiểu 1.5, 2.5
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
        ax.spines[spine].set_linewidth(1)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)


def _annotate_point(ax, x: int, y: float, text: str, total_epochs: int, above: bool):
    """Đặt nhãn cạnh 1 điểm, tự lật sang trái khi điểm nằm ở nửa phải biểu đồ
    để nhãn không tràn ra ngoài hoặc đè lên legend.
    """
    on_right_half = x > total_epochs / 2
    dx = -10 if on_right_half else 10
    ha = "right" if on_right_half else "left"
    dy = 14 if above else -30
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color=COLOR_TEXT, ha=ha,
    )


def plot_training_history(
    history: dict | pd.DataFrame,
    model_name: str,
    save_path: str | None = None,
    phase_split_epoch: int | None = None,
):
    """Vẽ 2 biểu đồ cạnh nhau: Accuracy và Loss (train vs validation).

    Args:
        history: dict Keras history.history, hoặc DataFrame đọc từ CSV log.
        model_name: tên model, dùng cho tiêu đề.
        save_path: nơi lưu file .png. Bỏ trống thì chỉ trả về figure.
        phase_split_epoch: nếu model train 2 giai đoạn (transfer learning:
            feature extraction -> fine-tuning), truyền số epoch của giai đoạn 1
            để vẽ 1 đường kẻ dọc phân tách, giúp thấy rõ tác động của fine-tuning.
    """
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history

    required = {"accuracy", "val_accuracy", "loss", "val_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"History thiếu các cột bắt buộc: {sorted(missing)}")

    epochs = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_SURFACE)
    for ax in axes:
        ax.set_facecolor(COLOR_SURFACE)

    # ---------------- Accuracy ----------------
    ax = axes[0]
    ax.plot(epochs, df["accuracy"], color=COLOR_TRAIN, linewidth=2, label="Train accuracy")
    ax.plot(epochs, df["val_accuracy"], color=COLOR_VAL, linewidth=2, label="Validation accuracy")

    # Đánh dấu epoch có val_accuracy cao nhất (điểm model được lưu checkpoint)
    best_idx = int(df["val_accuracy"].idxmax())
    best_val = float(df["val_accuracy"].max())
    ax.scatter([best_idx + 1], [best_val], color=COLOR_VAL, s=60, zorder=5,
               edgecolors=COLOR_SURFACE, linewidths=2)
    _annotate_point(
        ax, best_idx + 1, best_val,
        f"Tốt nhất: {best_val:.1%}\n(epoch {best_idx + 1})",
        len(df), above=False,
    )
    _style_axis(ax, "Training vs Validation Accuracy", "Epoch", "Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # ---------------- Loss ----------------
    ax = axes[1]
    ax.plot(epochs, df["loss"], color=COLOR_TRAIN, linewidth=2, label="Train loss")
    ax.plot(epochs, df["val_loss"], color=COLOR_VAL, linewidth=2, label="Validation loss")

    best_loss_idx = int(df["val_loss"].idxmin())
    best_loss = float(df["val_loss"].min())
    ax.scatter([best_loss_idx + 1], [best_loss], color=COLOR_VAL, s=60, zorder=5,
               edgecolors=COLOR_SURFACE, linewidths=2)
    _annotate_point(
        ax, best_loss_idx + 1, best_loss,
        f"Thấp nhất: {best_loss:.3f}\n(epoch {best_loss_idx + 1})",
        len(df), above=True,
    )
    _style_axis(ax, "Training vs Validation Loss", "Epoch", "Loss")
    # Legend đặt ở giữa-phải: loss thường giảm dần nên góc trên phải hay bị
    # đường dữ liệu hoặc nhãn điểm thấp nhất chiếm chỗ.
    ax.legend(frameon=False, fontsize=9, loc="center right")

    # Đường phân tách 2 giai đoạn của transfer learning
    if phase_split_epoch and 0 < phase_split_epoch < len(df):
        for ax in axes:
            ax.axvline(phase_split_epoch + 0.5, color=COLOR_MUTED, linestyle="--", linewidth=1)
            ax.text(
                phase_split_epoch + 0.7, ax.get_ylim()[1] * 0.97,
                "bắt đầu fine-tuning", fontsize=8, color=COLOR_MUTED, va="top",
            )

    fig.suptitle(f"Quá trình huấn luyện — {model_name}", fontsize=14, color=COLOR_TEXT)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=COLOR_SURFACE)
    return fig


def plot_from_csv(csv_path: str, model_name: str, save_path: str | None = None, phase_split_epoch: int | None = None):
    """Tiện ích: vẽ trực tiếp từ file CSV history đã lưu trong reports/logs/."""
    df = pd.read_csv(csv_path)
    return plot_training_history(df, model_name, save_path, phase_split_epoch)


def plot_all_available(config: dict = CONFIG):
    """Quét reports/logs/ và vẽ biểu đồ cho tất cả history CSV tìm được.

    Gọi tự động ở cuối quá trình training, hoặc chạy độc lập:
        python -m src.evaluation.plot_history
    """
    logs_dir = Path(config["paths"]["logs_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not logs_dir.exists():
        print(f"Chưa có thư mục log tại {logs_dir}. Hãy train model trước.")
        return []

    generated = []
    for csv_file in sorted(logs_dir.glob("*history*.csv")):
        model_name = csv_file.stem.replace("_history_combined", "").replace("_history", "")
        out_path = figures_dir / f"{model_name}_training_curves.png"

        split_epoch = None
        if "combined" in csv_file.stem:
            # History gộp của transfer learning -> lấy số epoch giai đoạn 1 từ metadata
            meta_path = Path(config["paths"]["reports_dir"]) / f"{model_name}_metadata.json"
            if meta_path.exists():
                import json
                with open(meta_path, encoding="utf-8") as f:
                    split_epoch = json.load(f).get("epochs_feature_extraction")

        try:
            plot_from_csv(str(csv_file), model_name, str(out_path), split_epoch)
            plt.close("all")
            generated.append(str(out_path))
            print(f"✓ Đã vẽ: {out_path}")
        except ValueError as e:
            print(f"✗ Bỏ qua {csv_file.name}: {e}")

    return generated


if __name__ == "__main__":
    plot_all_available()
