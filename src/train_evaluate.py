"""Train and evaluate a KNN classifier for Devanagari handwritten digits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import (
    DIGIT_GLYPHS,
    VALID_PREPROCESS_MODES,
    load_raw_digit_dataset,
    preprocess_image_array,
    preprocess_images,
)

VALID_WEIGHTS = {"uniform", "distance"}

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the KNN experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional single image size override (e.g. 16). Overrides --image-sizes.",
    )
    parser.add_argument(
        "--image-sizes",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Candidate square image sizes to compare on validation.",
    )
    parser.add_argument(
        "--preprocess-modes",
        type=str,
        nargs="+",
        default=["raw", "centered"],
        help="Preprocessing modes to compare on validation ('raw', 'centered').",
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        default=["distance", "uniform"],
        help="Neighbor voting weight strategies to compare ('distance', 'uniform').",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable exact duplicate removal before splitting (not recommended).",
    )
    return parser.parse_args(argv)


def create_stratified_splits(
    y: np.ndarray,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint stratified index arrays (train, validation, test)."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Test and validation sizes must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("Test size + validation size must be less than 1.")

    indices = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    validation_fraction = validation_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y[idx_trainval],
    )
    return idx_train, idx_val, idx_test


def select_best_configuration(
    feature_cache: dict[tuple[str, int], np.ndarray],
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    k_values: list[int],
    weights_list: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all (preprocess_mode, image_size, weights, K) candidates on validation only."""
    sorted_ks = sorted(set(k_values))
    if not sorted_ks or any(k < 1 for k in sorted_ks):
        raise ValueError("k_values must contain positive integers.")
    if any(k > len(idx_train) for k in sorted_ks):
        raise ValueError("One or more K values exceed the training sample count.")

    weights_options = list(dict.fromkeys(weights_list or ["distance", "uniform"]))
    for w in weights_options:
        if w not in VALID_WEIGHTS:
            raise ValueError(f"Unsupported weights option '{w}'. Expected one of {sorted(VALID_WEIGHTS)}.")

    y_train = y[idx_train]
    y_val = y[idx_val]
    rows: list[dict[str, Any]] = []

    for (mode, size), X_feat in feature_cache.items():
        X_train = X_feat[idx_train]
        X_val = X_feat[idx_val]
        for weight_mode in weights_options:
            for k in sorted_ks:
                t_fit_start = time.perf_counter()
                model = KNeighborsClassifier(
                    n_neighbors=k, weights=weight_mode, metric="euclidean", n_jobs=-1
                )
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - t_fit_start

                t_pred_start = time.perf_counter()
                val_pred = model.predict(X_val)
                val_pred_seconds = time.perf_counter() - t_pred_start

                rows.append({
                    "preprocess_mode": mode,
                    "image_size": int(size),
                    "feature_count": int(X_feat.shape[1]),
                    "weights": weight_mode,
                    "k": int(k),
                    "validation_accuracy": float(accuracy_score(y_val, val_pred)),
                    "fit_time_seconds": round(fit_seconds, 6),
                    "val_predict_time_seconds": round(val_pred_seconds, 6),
                })

    comparison_df = pd.DataFrame(rows).sort_values(
        ["validation_accuracy", "image_size", "k", "weights"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    best_row = comparison_df.iloc[0]
    selected = {
        "preprocess_mode": str(best_row["preprocess_mode"]),
        "image_size": int(best_row["image_size"]),
        "feature_count": int(best_row["feature_count"]),
        "weights": str(best_row["weights"]),
        "k": int(best_row["k"]),
        "validation_accuracy": float(best_row["validation_accuracy"]),
    }
    return comparison_df, selected


def _plot_confusion_matrix(cm: np.ndarray, best_cfg: dict[str, Any], test_acc: float, macro_f1: float, out_path: Path) -> None:
    """Save a publication-style confusion matrix heatmap PNG at 300 DPI."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    display_labels = [f"{i} ({DIGIT_GLYPHS[i]})" for i in range(10)]
    im = ax.imshow(cm, interpolation="nearest", cmap="PuRd", vmin=0, vmax=max(1, int(cm.max())))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Sample Count", fontsize=9, labelpad=6)

    thresh = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val == 0:
                ax.text(j, i, "0", ha="center", va="center", color="#cbd5e1", fontsize=8.5)
            else:
                text_color = "white" if val > thresh else "#1e293b"
                weight = "bold" if i == j else "bold"
                ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=9.5, fontweight=weight)

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=8.8)
    ax.set_yticklabels(display_labels, fontsize=8.8)
    ax.set_xlabel("Predicted Digit Class", fontsize=10, fontweight="bold", labelpad=6)
    ax.set_ylabel("True Digit Class", fontsize=10, fontweight="bold", labelpad=6)

    # Subtle cell gridlines for visual separation
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="#e2e8f0", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(
        f"Confusion Matrix ({best_cfg['preprocess_mode']} {best_cfg['image_size']}x{best_cfg['image_size']}, "
        f"K={best_cfg['k']}, {best_cfg['weights']})\n"
        f"Test Accuracy: {test_acc * 100:.2f}% | Macro F1: {macro_f1 * 100:.2f}%",
        fontsize=10.8,
        fontweight="bold",
        pad=10,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_k_validation_curve(comparison_df: pd.DataFrame, out_path: Path) -> None:
    """Plot clean academic validation accuracy curves across K, size, and weighting."""
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
    legend_handles = []
    legend_labels = []

    for ax_idx, (ax, mode) in enumerate(zip(axes_list, modes)):
        sub = comparison_df[comparison_df["preprocess_mode"] == mode].sort_values("k")
        for (size, w), grp in sub.groupby(["image_size", "weights"]):
            marker, ls, color, mfc = styles.get((int(size), str(w)), ("^", "-", "#475569", "#475569"))
            line, = ax.plot(
                grp["k"],
                grp["validation_accuracy"] * 100.0,
                marker=marker,
                linestyle=ls,
                color=color,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=1.4,
                linewidth=1.9,
                markersize=6.0,
                label=f"{size}x{size}, {w}",
            )
            if ax_idx == 0:
                legend_handles.append(line)
                legend_labels.append(f"{size}x{size} ({w})")

        if mode == best_row["preprocess_mode"]:
            bx = int(best_row["k"])
            by = float(best_row["validation_accuracy"]) * 100.0
            ax.annotate(
                f"Selected: K={bx}, {by:.2f}%",
                xy=(bx, by),
                xytext=(bx + 1.3, by - 5.5),
                fontsize=8.5,
                fontweight="bold",
                color="#a8245c",
                arrowprops={"arrowstyle": "->", "color": "#a8245c", "lw": 1.1},
                bbox={"boxstyle": "round,pad=0.25", "fc": "#fdf2f7", "ec": "#a8245c", "lw": 0.7},
            )

        ax.set_title(f"Preprocessing: {mode}", fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xlabel("Number of Neighbors (K)", fontsize=9.5)
        ax.set_xticks(sorted(comparison_df["k"].unique()))
        ax.set_ylim(53, 102)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8.8)

    axes_list[0].set_ylabel("Validation Accuracy (%)", fontsize=9.5, fontweight="bold")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Validation Accuracy vs. K Across Preprocessing, Resolution, and Weighting",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_class_distribution(data_meta: dict[str, Any], out_path: Path) -> None:
    """Plot raw vs. deduplicated unique image counts for classes 0-9."""
    digits = [str(i) for i in range(10)]
    labels = [f"{i} ({DIGIT_GLYPHS[i]})" for i in range(10)]
    raw_counts = [data_meta["raw_class_counts"][d] for d in digits]
    unique_counts = [data_meta["class_counts"][d] for d in digits]

    x = np.arange(len(digits))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars_raw = ax.bar(
        x - width / 2, raw_counts, width, label=f"Raw CSV Rows (n={sum(raw_counts):,})",
        color="#94a3b8", edgecolor="#475569", linewidth=0.6,
    )
    bars_uniq = ax.bar(
        x + width / 2, unique_counts, width, label=f"Unique Images (n={sum(unique_counts):,})",
        color="#a8245c", edgecolor="#6b1439", linewidth=0.6,
    )

    for bar in bars_raw:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=7.8, color="#475569",
        )
    for bar in bars_uniq:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color="#831843",
        )

    ax.set_title("Devanagari Digit Class Distribution: Raw Rows vs. Deduplicated Unique Images", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Digit Class", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Image Count", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(0, max(raw_counts) * 1.14)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_dataset_samples(
    raw_images: list[np.ndarray],
    y: np.ndarray,
    out_path: Path,
    samples_per_class: int = 5,
) -> None:
    """Save a crisp 10x5 grid of representative original 65x65 digit images."""
    fig, axes = plt.subplots(10, samples_per_class, figsize=(samples_per_class * 1.15 + 0.7, 10.2))
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        step = max(1, len(digit_indices) // samples_per_class)
        chosen = digit_indices[::step][:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[digit, col]
            if col < len(chosen):
                ax.imshow(raw_images[int(chosen[col])], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cbd5e1")
                spine.set_linewidth(0.7)
            if col == 0:
                ax.set_ylabel(
                    f"{digit} ({DIGIT_GLYPHS[digit]})",
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=22,
                    va="center",
                    color="#1e293b",
                )
    fig.suptitle(
        "Representative Unique Devanagari Digit Images (65x65 Grayscale)",
        fontsize=11.2,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.10)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_preprocessing_comparison(
    raw_images: list[np.ndarray],
    y: np.ndarray,
    out_path: Path,
) -> None:
    """Plot side-by-side comparison of Original 65x65, Raw 16x16, and Centered 16x16."""
    sample_digits = [0, 2, 4, 6, 7, 9]
    fig, axes = plt.subplots(3, len(sample_digits), figsize=(len(sample_digits) * 1.45 + 1.1, 4.7))
    row_titles = ["Original (65x65)", "Raw Resize (16x16)", "Centered (16x16)"]

    for col, digit in enumerate(sample_digits):
        idx = int(np.where(y == digit)[0][0])
        orig = raw_images[idx]
        raw_16 = preprocess_image_array(orig, image_size=16, preprocess_mode="raw").reshape(16, 16)
        cen_16 = preprocess_image_array(orig, image_size=16, preprocess_mode="centered").reshape(16, 16)

        panels = [(orig, 255.0), (raw_16, 1.0), (cen_16, 1.0)]
        for row, (arr, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#94a3b8" if row < 2 else "#a8245c")
                spine.set_linewidth(0.8 if row < 2 else 1.2)
            if row == 0:
                ax.set_title(f"Digit {digit} ({DIGIT_GLYPHS[digit]})", fontsize=9.5, fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=8.8, fontweight="bold", rotation=0, labelpad=48, va="center")

    fig.suptitle(
        "Effect of Bounding-Box Centering on Low-Resolution Feature Alignment",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(wspace=0.10, hspace=0.14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_prediction_samples(
    raw_images: list[np.ndarray],
    idx_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    nn_dists: np.ndarray,
    out_path: Path,
) -> None:
    """Plot a crisp grid showing representative misclassified and correct test predictions."""
    correct_pos = np.where(y_test == y_pred)[0]
    wrong_pos = np.where(y_test != y_pred)[0]

    selected_positions: list[tuple[int, bool]] = []
    for p in wrong_pos[:5]:
        selected_positions.append((int(p), False))
    needed = 15 - len(selected_positions)
    if len(correct_pos) > 0:
        stride = max(1, len(correct_pos) // max(1, needed))
        for p in correct_pos[::stride][:needed]:
            selected_positions.append((int(p), True))

    n_items = len(selected_positions)
    cols = 5
    rows = max(1, int(np.ceil(n_items / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(10.2, rows * 2.25))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= n_items:
            ax.axis("off")
            continue
        pos, is_ok = selected_positions[ax_idx]
        raw_idx = int(idx_test[pos])
        true_lbl = int(y_test[pos])
        pred_lbl = int(y_pred[pos])
        dist = float(nn_dists[pos])
        ax.imshow(raw_images[raw_idx], cmap="gray", vmin=0, vmax=255)
        border_color = "#15803d" if is_ok else "#a8245c"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.5 if not is_ok else 1.0)
        tag = "Correct" if is_ok else "Error"
        ax.set_title(
            f"{tag} | True: {true_lbl} ({DIGIT_GLYPHS[true_lbl]}), Pred: {pred_lbl} ({DIGIT_GLYPHS[pred_lbl]})\n1-NN d={dist:.2f}",
            fontsize=8.0,
            color=border_color,
            fontweight="bold" if not is_ok else "normal",
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Held-Out Test Predictions: Misclassified ({len(wrong_pos)} total) & Correct Samples",
        fontsize=11.2,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(wspace=0.12, hspace=0.34)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    image_sizes: list[int] | None = None,
    preprocess_modes: list[str] | None = None,
    k_values: list[int] | None = None,
    weights_list: list[str] | None = None,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """Run full loading, validation selection, refit, test evaluation, and artifact generation."""
    sizes = sorted(set(image_sizes or [16, 32]))
    modes = list(dict.fromkeys(preprocess_modes or ["raw", "centered"]))
    ks = sorted(set(k_values or [1, 3, 5, 7, 9]))
    weights_options = list(dict.fromkeys(weights_list or ["distance", "uniform"]))

    if any(s < 8 for s in sizes):
        raise ValueError("All image sizes must be at least 8 pixels.")
    for mode in modes:
        if mode not in VALID_PREPROCESS_MODES:
            raise ValueError(f"Invalid preprocess mode '{mode}'.")
    for w in weights_options:
        if w not in VALID_WEIGHTS:
            raise ValueError(f"Invalid weights mode '{w}'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_load_start = time.perf_counter()
    raw_images, y, data_meta = load_raw_digit_dataset(data_dir, deduplicate=deduplicate)
    load_seconds = time.perf_counter() - t_load_start

    idx_train, idx_val, idx_test = create_stratified_splits(
        y, test_size=test_size, validation_size=validation_size, random_state=random_state
    )
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0

    feature_cache: dict[tuple[str, int], np.ndarray] = {}
    for mode in modes:
        for size in sizes:
            feature_cache[(mode, size)] = preprocess_images(
                raw_images, image_size=size, preprocess_mode=mode
            )

    t_search_start = time.perf_counter()
    comparison_df, best_cfg = select_best_configuration(
        feature_cache, y, idx_train, idx_val, ks, weights_list=weights_options
    )
    val_search_seconds = time.perf_counter() - t_search_start
    comparison_df.to_csv(output_dir / "k_comparison.csv", index=False)

    X_best = feature_cache[(best_cfg["preprocess_mode"], best_cfg["image_size"])]
    X_train, y_train = X_best[idx_train], y[idx_train]
    X_val, y_val = X_best[idx_val], y[idx_val]
    X_test, y_test = X_best[idx_test], y[idx_test]

    X_final_train = np.concatenate([X_train, X_val], axis=0)
    y_final_train = np.concatenate([y_train, y_val], axis=0)

    t_fit_start = time.perf_counter()
    final_model = KNeighborsClassifier(
        n_neighbors=best_cfg["k"],
        weights=best_cfg["weights"],
        metric="euclidean",
        n_jobs=-1,
    )
    final_model.fit(X_final_train, y_final_train)
    final_fit_seconds = time.perf_counter() - t_fit_start

    t_test_start = time.perf_counter()
    y_pred = final_model.predict(X_test)
    test_inference_seconds = time.perf_counter() - t_test_start

    nn_dists, _ = final_model.kneighbors(X_test, n_neighbors=1)

    labels = list(range(10))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    target_names = [f"{i} ({DIGIT_GLYPHS[i]})" for i in labels]
    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    per_class_metrics = {
        str(lbl): {
            "glyph": DIGIT_GLYPHS[lbl],
            "precision": float(prec_arr[lbl]),
            "recall": float(rec_arr[lbl]),
            "f1_score": float(f1_arr[lbl]),
            "support": int(sup_arr[lbl]),
        }
        for lbl in labels
    }

    report_rows = [
        {
            "digit": lbl,
            "glyph": DIGIT_GLYPHS[lbl],
            "precision": round(float(prec_arr[lbl]), 6),
            "recall": round(float(rec_arr[lbl]), 6),
            "f1_score": round(float(f1_arr[lbl]), 6),
            "support": int(sup_arr[lbl]),
        }
        for lbl in labels
    ]
    pd.DataFrame(report_rows).to_csv(output_dir / "classification_report.csv", index=False)

    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        output_dir / "confusion_matrix.csv", index=True
    )

    test_acc = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    _plot_confusion_matrix(cm, best_cfg, test_acc, test_macro_f1, output_dir / "confusion_matrix.png")
    _plot_k_validation_curve(comparison_df, output_dir / "k_validation_curve.png")
    _plot_class_distribution(data_meta, output_dir / "class_distribution.png")
    _plot_dataset_samples(raw_images, y, output_dir / "dataset_samples.png")
    _plot_preprocessing_comparison(raw_images, y, output_dir / "preprocessing_comparison.png")
    _plot_prediction_samples(raw_images, idx_test, y_test, y_pred, nn_dists[:, 0], output_dir / "prediction_samples.png")

    sample_ids = data_meta["sample_paths"]
    prediction_df = pd.DataFrame({
        "test_index": idx_test.astype(int),
        "sample_source": [sample_ids[int(i)] for i in idx_test],
        "actual_label": y_test.astype(int),
        "predicted_label": y_pred.astype(int),
        "actual_glyph": [DIGIT_GLYPHS[int(v)] for v in y_test],
        "predicted_glyph": [DIGIT_GLYPHS[int(v)] for v in y_pred],
        "correct": (y_test == y_pred),
        "nearest_neighbor_distance": np.round(nn_dists[:, 0], 6),
    })
    prediction_df.to_csv(output_dir / "sample_predictions.csv", index=False)

    inspection_summary = {
        "digits_directory": data_meta["digits_directory"],
        "source_format": data_meta["source_format"],
        "source_directory": data_meta["source_directory"],
        "detected_dimensions": data_meta["detected_dimensions"],
        "detected_formats": data_meta["detected_formats"],
        "raw_digit_images": data_meta["raw_image_count"],
        "duplicates_removed": data_meta["duplicates_removed"],
        "cross_class_duplicates": data_meta["cross_class_duplicates"],
        "unique_digit_images": data_meta["image_count"],
        "raw_class_counts": data_meta["raw_class_counts"],
        "unique_class_counts": data_meta["class_counts"],
        "ignored_files_or_folders": data_meta["ignored_class_folders"],
        "corrupt_files": data_meta["corrupt_files"],
        "corrupt_rows_skipped": data_meta["corrupt_rows_skipped"],
        "excluded_alphabet_subset": data_meta["excluded_alphabet_subset"],
    }
    (output_dir / "dataset_inspection.json").write_text(
        json.dumps(inspection_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics: dict[str, Any] = {
        "project": "Devanagari Handwritten Digit Recognition Using KNN",
        "dataset_source": "https://www.kaggle.com/datasets/sanjaypate/devanagari-script-digital-44000-images-dataset",
        "dataset_doi": "10.34740/KAGGLE/DSV/9820160",
        "dataset_directory": data_meta["digits_directory"],
        "source_format": data_meta["source_format"],
        "raw_digit_images": data_meta["raw_image_count"],
        "duplicates_removed": data_meta["duplicates_removed"],
        "total_images_loaded": data_meta["image_count"],
        "class_counts": data_meta["class_counts"],
        "raw_class_counts": data_meta["raw_class_counts"],
        "excluded_alphabet_rows": data_meta["excluded_alphabet_subset"].get("rows", 0),
        "split_counts": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "final_training_after_refit": int(len(y_final_train)),
            "test": int(len(y_test)),
        },
        "distance_metric": "euclidean",
        "weights": best_cfg["weights"],
        "selected_preprocess_mode": best_cfg["preprocess_mode"],
        "image_size": best_cfg["image_size"],
        "features_per_image": best_cfg["feature_count"],
        "selected_k": best_cfg["k"],
        "selected_validation_accuracy": best_cfg["validation_accuracy"],
        "candidate_comparisons": comparison_df.to_dict(orient="records"),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_weighted_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class_metrics": per_class_metrics,
        "timing_seconds": {
            "data_loading": round(load_seconds, 4),
            "validation_search": round(val_search_seconds, 4),
            "final_model_fit": round(final_fit_seconds, 4),
            "test_inference": round(test_inference_seconds, 4),
        },
        "random_state": int(random_state),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "pillow_version": PIL.__version__,
            "scikit_learn_version": sklearn.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for training and evaluating the KNN Devanagari digit model."""
    args = parse_args(argv)
    image_sizes = [args.image_size] if args.image_size is not None else args.image_sizes
    metrics = run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_sizes=image_sizes,
        preprocess_modes=args.preprocess_modes,
        k_values=args.k_values,
        weights_list=args.weights,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        deduplicate=not args.no_deduplicate,
    )
    print(json.dumps({
        "total_images_loaded": metrics["total_images_loaded"],
        "split_counts": metrics["split_counts"],
        "selected_preprocess_mode": metrics["selected_preprocess_mode"],
        "image_size": metrics["image_size"],
        "weights": metrics["weights"],
        "selected_k": metrics["selected_k"],
        "selected_validation_accuracy": metrics["selected_validation_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "test_macro_precision": metrics["test_macro_precision"],
        "test_macro_recall": metrics["test_macro_recall"],
        "test_macro_f1": metrics["test_macro_f1"],
        "timing_seconds": metrics["timing_seconds"],
    }, indent=2))
    print("\nClassification report:\n")
    print((Path(args.output_dir) / "classification_report.txt").read_text(encoding="utf-8"))
    print(f"Saved evaluation outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
