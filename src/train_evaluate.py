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

from src.data_loader import DIGIT_GLYPHS, VALID_PREPROCESS_MODES, load_raw_digit_dataset, preprocess_images
from src.plots import generate_all_plots

VALID_WEIGHTS = {"uniform", "distance"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the KNN experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--image-size", type=int, default=None, help="Single image size override.")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--preprocess-modes", type=str, nargs="+", default=["raw", "centered"])
    parser.add_argument("--weights", type=str, nargs="+", default=["distance", "uniform"])
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser.parse_args(argv)


def create_stratified_splits(
    y: np.ndarray,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint stratified index arrays (train, validation, test)."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("Test and validation sizes must lie in (0, 1) and sum to < 1.")

    indices = np.arange(len(y))
    idx_tv, idx_test = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y)
    val_frac = validation_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(idx_tv, test_size=val_frac, random_state=random_state, stratify=y[idx_tv])
    return idx_train, idx_val, idx_test


def select_best_configuration(
    feature_cache: dict[tuple[str, int], np.ndarray],
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    k_values: list[int],
    weights_list: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all (preprocess_mode, image_size, weights, K) candidates on validation."""
    ks = sorted(set(k_values))
    if not ks or any(k < 1 or k > len(idx_train) for k in ks):
        raise ValueError("k_values must contain positive integers <= training sample count.")
    weights_options = list(dict.fromkeys(weights_list or ["distance", "uniform"]))
    if any(w not in VALID_WEIGHTS for w in weights_options):
        raise ValueError(f"Unsupported weights in {weights_options}.")

    rows = []
    y_train, y_val = y[idx_train], y[idx_val]
    for (mode, size), X_feat in feature_cache.items():
        X_train, X_val = X_feat[idx_train], X_feat[idx_val]
        for weight_mode in weights_options:
            for k in ks:
                t0 = time.perf_counter()
                model = KNeighborsClassifier(n_neighbors=k, weights=weight_mode, metric="euclidean", n_jobs=-1).fit(X_train, y_train)
                fit_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                val_pred = model.predict(X_val)
                pred_s = time.perf_counter() - t1
                rows.append({
                    "preprocess_mode": mode, "image_size": int(size), "feature_count": int(X_feat.shape[1]),
                    "weights": weight_mode, "k": int(k), "validation_accuracy": float(accuracy_score(y_val, val_pred)),
                    "fit_time_seconds": round(fit_s, 6), "val_predict_time_seconds": round(pred_s, 6),
                })

    comp_df = pd.DataFrame(rows).sort_values(
        ["validation_accuracy", "image_size", "k", "weights"], ascending=[False, True, True, True]
    ).reset_index(drop=True)
    top = comp_df.iloc[0]
    best = {
        "preprocess_mode": str(top["preprocess_mode"]), "image_size": int(top["image_size"]),
        "feature_count": int(top["feature_count"]), "weights": str(top["weights"]),
        "k": int(top["k"]), "validation_accuracy": float(top["validation_accuracy"]),
    }
    return comp_df, best


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
    """Run data loading, validation selection, final refit, test evaluation, and artifact export."""
    sizes = sorted(set(image_sizes or [16, 32]))
    modes = list(dict.fromkeys(preprocess_modes or ["raw", "centered"]))
    ks = sorted(set(k_values or [1, 3, 5, 7, 9]))
    weights_options = list(dict.fromkeys(weights_list or ["distance", "uniform"]))
    if any(s < 8 for s in sizes) or any(m not in VALID_PREPROCESS_MODES for m in modes) or any(w not in VALID_WEIGHTS for w in weights_options):
        raise ValueError("Invalid image size, preprocess mode, or weights configuration.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    raw_images, y, meta = load_raw_digit_dataset(data_dir, deduplicate=deduplicate)
    load_s = time.perf_counter() - t_load

    idx_train, idx_val, idx_test = create_stratified_splits(y, test_size, validation_size, random_state)
    cache = {(m, s): preprocess_images(raw_images, s, m) for m in modes for s in sizes}

    t_val = time.perf_counter()
    comp_df, best = select_best_configuration(cache, y, idx_train, idx_val, ks, weights_options)
    val_s = time.perf_counter() - t_val
    comp_df.to_csv(output_dir / "k_comparison.csv", index=False)

    X_best = cache[(best["preprocess_mode"], best["image_size"])]
    X_refit = np.concatenate([X_best[idx_train], X_best[idx_val]], axis=0)
    y_refit = np.concatenate([y[idx_train], y[idx_val]], axis=0)
    X_test, y_test = X_best[idx_test], y[idx_test]

    t_fit = time.perf_counter()
    final_model = KNeighborsClassifier(n_neighbors=best["k"], weights=best["weights"], metric="euclidean", n_jobs=-1).fit(X_refit, y_refit)
    fit_s = time.perf_counter() - t_fit

    t_inf = time.perf_counter()
    y_pred = final_model.predict(X_test)
    inf_s = time.perf_counter() - t_inf
    nn_dists, _ = final_model.kneighbors(X_test, n_neighbors=1)

    labels = list(range(10))
    target_names = [f"{i} ({DIGIT_GLYPHS[i]})" for i in labels]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_txt = classification_report(y_test, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(output_dir / "confusion_matrix.csv")

    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    pd.DataFrame([
        {"digit": d, "glyph": DIGIT_GLYPHS[d], "precision": round(float(prec[d]), 6),
         "recall": round(float(rec[d]), 6), "f1_score": round(float(f1[d]), 6), "support": int(sup[d])}
        for d in labels
    ]).to_csv(output_dir / "classification_report.csv", index=False)

    test_acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    generate_all_plots(output_dir, raw_images, y, meta, comp_df, best, cm, test_acc, macro_f1, idx_test, y_test, y_pred, nn_dists[:, 0])

    pd.DataFrame({
        "test_index": idx_test.astype(int),
        "sample_source": [meta["sample_paths"][int(i)] for i in idx_test],
        "actual_label": y_test.astype(int), "predicted_label": y_pred.astype(int),
        "actual_glyph": [DIGIT_GLYPHS[int(v)] for v in y_test], "predicted_glyph": [DIGIT_GLYPHS[int(v)] for v in y_pred],
        "correct": y_test == y_pred, "nearest_neighbor_distance": np.round(nn_dists[:, 0], 6),
    }).to_csv(output_dir / "sample_predictions.csv", index=False)

    inspection = {
        "digits_directory": meta["digits_directory"], "source_format": meta["source_format"],
        "source_directory": meta["source_directory"], "detected_dimensions": meta["detected_dimensions"],
        "detected_formats": meta["detected_formats"], "raw_digit_images": meta["raw_image_count"],
        "duplicates_removed": meta["duplicates_removed"], "cross_class_duplicates": meta["cross_class_duplicates"],
        "unique_digit_images": meta["image_count"], "raw_class_counts": meta["raw_class_counts"],
        "unique_class_counts": meta["class_counts"], "ignored_files_or_folders": meta["ignored_class_folders"],
        "corrupt_files": meta["corrupt_files"], "corrupt_rows_skipped": meta["corrupt_rows_skipped"],
        "excluded_alphabet_subset": meta["excluded_alphabet_subset"],
    }
    (output_dir / "dataset_inspection.json").write_text(json.dumps(inspection, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics: dict[str, Any] = {
        "project": "Devanagari Handwritten Digit Recognition Using KNN",
        "dataset_source": "https://www.kaggle.com/datasets/sanjaypate/devanagari-script-digital-44000-images-dataset",
        "dataset_doi": "10.34740/KAGGLE/DSV/9820160",
        "dataset_directory": meta["digits_directory"], "source_format": meta["source_format"],
        "raw_digit_images": meta["raw_image_count"], "duplicates_removed": meta["duplicates_removed"],
        "total_images_loaded": meta["image_count"], "class_counts": meta["class_counts"],
        "raw_class_counts": meta["raw_class_counts"],
        "excluded_alphabet_rows": meta["excluded_alphabet_subset"].get("rows", 0),
        "split_counts": {"train": len(idx_train), "validation": len(idx_val), "final_training_after_refit": len(y_refit), "test": len(idx_test)},
        "distance_metric": "euclidean", "weights": best["weights"],
        "selected_preprocess_mode": best["preprocess_mode"], "image_size": best["image_size"],
        "features_per_image": best["feature_count"], "selected_k": best["k"],
        "selected_validation_accuracy": best["validation_accuracy"],
        "candidate_comparisons": comp_df.to_dict(orient="records"),
        "test_accuracy": test_acc,
        "test_macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_f1": macro_f1,
        "test_weighted_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class_metrics": {
            str(d): {"glyph": DIGIT_GLYPHS[d], "precision": float(prec[d]), "recall": float(rec[d]), "f1_score": float(f1[d]), "support": int(sup[d])}
            for d in labels
        },
        "timing_seconds": {"data_loading": round(load_s, 4), "validation_search": round(val_s, 4), "final_model_fit": round(fit_s, 4), "test_inference": round(inf_s, 4)},
        "random_state": int(random_state),
        "environment": {
            "python_version": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy_version": np.__version__, "pandas_version": pd.__version__,
            "pillow_version": PIL.__version__, "scikit_learn_version": sklearn.__version__, "matplotlib_version": matplotlib.__version__,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for training and evaluating the KNN model."""
    args = parse_args(argv)
    sizes = [args.image_size] if args.image_size is not None else args.image_sizes
    metrics = run_experiment(
        args.data_dir, args.output_dir, sizes, args.preprocess_modes, args.k_values,
        args.weights, args.test_size, args.validation_size, args.random_state, not args.no_deduplicate,
    )
    print(json.dumps({k: metrics[k] for k in (
        "total_images_loaded", "split_counts", "selected_preprocess_mode", "image_size", "weights",
        "selected_k", "selected_validation_accuracy", "test_accuracy", "test_macro_precision",
        "test_macro_recall", "test_macro_f1", "timing_seconds",
    )}, indent=2))
    print("\nClassification report:\n")
    print((Path(args.output_dir) / "classification_report.txt").read_text(encoding="utf-8"))
    print(f"Saved evaluation outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
