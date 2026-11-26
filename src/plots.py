"""Generate 300 DPI academic figures for the Devanagari KNN experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import DIGIT_GLYPHS, preprocess_image_array

plt.rcParams.update({
    "font.sans-serif": ["Nirmala UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def _despine(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Apply consistent minimal academic axis styling."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, linestyle="--", alpha=0.45, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.8)


def _frame_image_axes(ax: plt.Axes, color: str = "#cbd5e1", lw: float = 0.7) -> None:
    """Remove ticks and apply subtle border to an image subplot."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def plot_confusion_matrix(
    cm: np.ndarray,
    best_cfg: dict[str, Any],
    test_acc: float,
    macro_f1: float,
    out_path: Path,
) -> None:
    """Save a 300 DPI confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    labels = [f"{i} ({DIGIT_GLYPHS[i]})" for i in range(10)]
    im = ax.imshow(cm, interpolation="nearest", cmap="PuRd", vmin=0, vmax=max(1, int(cm.max())))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Sample Count", fontsize=9, labelpad=6)

    thresh = cm.max() * 0.55
    for i in range(10):
        for j in range(10):
            val = int(cm[i, j])
            color = "#cbd5e1" if val == 0 else ("white" if val > thresh else "#1e293b")
            ax.text(j, i, str(val), ha="center", va="center", color=color,
                    fontsize=8.5 if val == 0 else 9.5, fontweight="bold" if val > 0 else "normal")

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.8)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlabel("Predicted Digit Class", fontsize=10, fontweight="bold", labelpad=6)
    ax.set_ylabel("True Digit Class", fontsize=10, fontweight="bold", labelpad=6)
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="#e2e8f0", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(
        f"Confusion Matrix ({best_cfg['preprocess_mode']} {best_cfg['image_size']}x{best_cfg['image_size']}, "
        f"K={best_cfg['k']}, {best_cfg['weights']})\nTest Accuracy: {test_acc * 100:.2f}% | Macro F1: {macro_f1 * 100:.2f}%",
        fontsize=10.8, fontweight="bold", pad=10,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_k_validation_curve(comparison_df: pd.DataFrame, out_path: Path) -> None:
    """Plot validation accuracy across K, preprocessing mode, image size, and weights."""
    modes = list(dict.fromkeys(comparison_df["preprocess_mode"].tolist()))
    fig, axes = plt.subplots(1, len(modes), figsize=(5.3 * len(modes), 4.3), sharey=True)
    axes_list = np.atleast_1d(axes).ravel()
    styles = {
        (16, "distance"): ("o", "-", "#a8245c", "#a8245c"),
        (16, "uniform"): ("o", "--", "#d94884", "white"),
        (32, "distance"): ("s", "-", "#1e293b", "#1e293b"),
        (32, "uniform"): ("s", "--", "#64748b", "white"),
    }
    best_row = comparison_df.iloc[0]
    handles, legend_labels = [], []

    for idx, (ax, mode) in enumerate(zip(axes_list, modes)):
        sub = comparison_df[comparison_df["preprocess_mode"] == mode].sort_values("k")
        for (size, w), grp in sub.groupby(["image_size", "weights"]):
            marker, ls, color, mfc = styles.get((int(size), str(w)), ("^", "-", "#475569", "#475569"))
            line, = ax.plot(
                grp["k"], grp["validation_accuracy"] * 100.0, marker=marker, linestyle=ls,
                color=color, markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.4,
                linewidth=1.9, markersize=6.0,
            )
            if idx == 0:
                handles.append(line)
                legend_labels.append(f"{size}x{size} ({w})")

        if mode == best_row["preprocess_mode"]:
            bx, by = int(best_row["k"]), float(best_row["validation_accuracy"]) * 100.0
            ax.annotate(
                f"Selected: K={bx}, {by:.2f}%", xy=(bx, by), xytext=(bx + 1.3, by - 5.5),
                fontsize=8.5, fontweight="bold", color="#a8245c",
                arrowprops={"arrowstyle": "->", "color": "#a8245c", "lw": 1.1},
                bbox={"boxstyle": "round,pad=0.25", "fc": "#fdf2f7", "ec": "#a8245c", "lw": 0.7},
            )
        ax.set_title(f"Preprocessing: {mode}", fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xlabel("Number of Neighbors (K)", fontsize=9.5)
        ax.set_xticks(sorted(comparison_df["k"].unique()))
        ax.set_ylim(53, 102)
        _despine(ax, "both")

    axes_list[0].set_ylabel("Validation Accuracy (%)", fontsize=9.5, fontweight="bold")
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=True,
               facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Validation Accuracy vs. K Across Preprocessing, Resolution, and Weighting",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_class_distribution(data_meta: dict[str, Any], out_path: Path) -> None:
    """Plot raw CSV rows vs. deduplicated unique image counts for classes 0-9."""
    digits = [str(i) for i in range(10)]
    raw_counts = [data_meta["raw_class_counts"][d] for d in digits]
    uniq_counts = [data_meta["class_counts"][d] for d in digits]
    x, width = np.arange(10), 0.38

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars_raw = ax.bar(x - width / 2, raw_counts, width, label=f"Raw CSV Rows (n={sum(raw_counts):,})",
                      color="#94a3b8", edgecolor="#475569", linewidth=0.6)
    bars_uniq = ax.bar(x + width / 2, uniq_counts, width, label=f"Unique Images (n={sum(uniq_counts):,})",
                       color="#a8245c", edgecolor="#6b1439", linewidth=0.6)

    for bars, color, fw in [(bars_raw, "#475569", "normal"), (bars_uniq, "#831843", "bold")]:
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.9, color=color, fontweight=fw)

    ax.set_title("Devanagari Digit Class Distribution: Raw Rows vs. Deduplicated Unique Images",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Digit Class", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Image Count", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} ({DIGIT_GLYPHS[i]})" for i in range(10)], fontsize=8.8)
    ax.set_ylim(0, max(raw_counts) * 1.14)
    _despine(ax, "y")
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dataset_samples(raw_images: list[np.ndarray], y: np.ndarray, out_path: Path, samples_per_class: int = 5) -> None:
    """Save a 10x5 grid of representative original 65x65 digit images."""
    fig, axes = plt.subplots(10, samples_per_class, figsize=(samples_per_class * 1.15 + 0.7, 10.2))
    for digit in range(10):
        indices = np.where(y == digit)[0]
        chosen = indices[::max(1, len(indices) // samples_per_class)][:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[digit, col]
            if col < len(chosen):
                ax.imshow(raw_images[int(chosen[col])], cmap="gray", vmin=0, vmax=255)
            _frame_image_axes(ax)
            if col == 0:
                ax.set_ylabel(f"{digit} ({DIGIT_GLYPHS[digit]})", fontsize=9.5, fontweight="bold",
                              rotation=0, labelpad=22, va="center", color="#1e293b")
    fig.suptitle("Representative Unique Devanagari Digit Images (65x65 Grayscale)", fontsize=11.2, fontweight="bold", y=0.995)
    fig.subplots_adjust(wspace=0.08, hspace=0.10)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_preprocessing_comparison(raw_images: list[np.ndarray], y: np.ndarray, out_path: Path) -> None:
    """Compare Original 65x65, Raw Resize 16x16, and Centered 16x16 across sample digits."""
    sample_digits = [0, 2, 4, 6, 7, 9]
    row_titles = ["Original (65x65)", "Raw Resize (16x16)", "Centered (16x16)"]
    fig, axes = plt.subplots(3, len(sample_digits), figsize=(len(sample_digits) * 1.45 + 1.1, 4.7))

    for col, digit in enumerate(sample_digits):
        orig = raw_images[int(np.where(y == digit)[0][0])]
        panels = [
            (orig, 255.0),
            (preprocess_image_array(orig, 16, "raw").reshape(16, 16), 1.0),
            (preprocess_image_array(orig, 16, "centered").reshape(16, 16), 1.0),
        ]
        for row, (arr, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
            _frame_image_axes(ax, "#94a3b8" if row < 2 else "#a8245c", 0.8 if row < 2 else 1.2)
            if row == 0:
                ax.set_title(f"Digit {digit} ({DIGIT_GLYPHS[digit]})", fontsize=9.5, fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=8.8, fontweight="bold", rotation=0, labelpad=48, va="center")

    fig.suptitle("Effect of Bounding-Box Centering on Low-Resolution Feature Alignment", fontsize=11, fontweight="bold", y=0.99)
    fig.subplots_adjust(wspace=0.10, hspace=0.14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_prediction_samples(
    raw_images: list[np.ndarray],
    idx_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    nn_dists: np.ndarray,
    out_path: Path,
) -> None:
    """Plot a grid of misclassified and representative correct test predictions."""
    wrong_pos = np.where(y_test != y_pred)[0]
    correct_pos = np.where(y_test == y_pred)[0]
    picks = [(int(p), False) for p in wrong_pos[:5]]
    needed = 15 - len(picks)
    if len(correct_pos) > 0:
        picks.extend((int(p), True) for p in correct_pos[::max(1, len(correct_pos) // max(1, needed))][:needed])

    rows = max(1, int(np.ceil(len(picks) / 5)))
    fig, axes = plt.subplots(rows, 5, figsize=(10.2, rows * 2.25))
    for ax_idx, ax in enumerate(np.atleast_1d(axes).ravel()):
        if ax_idx >= len(picks):
            ax.axis("off")
            continue
        pos, is_ok = picks[ax_idx]
        t_lbl, p_lbl = int(y_test[pos]), int(y_pred[pos])
        color = "#15803d" if is_ok else "#a8245c"
        ax.imshow(raw_images[int(idx_test[pos])], cmap="gray", vmin=0, vmax=255)
        _frame_image_axes(ax, color, 1.0 if is_ok else 1.5)
        ax.set_title(
            f"{'Correct' if is_ok else 'Error'} | True: {t_lbl} ({DIGIT_GLYPHS[t_lbl]}), "
            f"Pred: {p_lbl} ({DIGIT_GLYPHS[p_lbl]})\n1-NN d={float(nn_dists[pos]):.2f}",
            fontsize=8.0, color=color, fontweight="normal" if is_ok else "bold", pad=4,
        )

    fig.suptitle(f"Held-Out Test Predictions: Misclassified ({len(wrong_pos)} total) & Correct Samples",
                 fontsize=11.2, fontweight="bold", y=0.99)
    fig.subplots_adjust(wspace=0.12, hspace=0.34)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_all_plots(
    output_dir: Path,
    raw_images: list[np.ndarray],
    y: np.ndarray,
    data_meta: dict[str, Any],
    comparison_df: pd.DataFrame,
    best_cfg: dict[str, Any],
    cm: np.ndarray,
    test_acc: float,
    macro_f1: float,
    idx_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    nn_dists: np.ndarray,
) -> None:
    """Render and save all 6 evaluation plots into output_dir."""
    plot_confusion_matrix(cm, best_cfg, test_acc, macro_f1, output_dir / "confusion_matrix.png")
    plot_k_validation_curve(comparison_df, output_dir / "k_validation_curve.png")
    plot_class_distribution(data_meta, output_dir / "class_distribution.png")
    plot_dataset_samples(raw_images, y, output_dir / "dataset_samples.png")
    plot_preprocessing_comparison(raw_images, y, output_dir / "preprocessing_comparison.png")
    plot_prediction_samples(raw_images, idx_test, y_test, y_pred, nn_dists, output_dir / "prediction_samples.png")
