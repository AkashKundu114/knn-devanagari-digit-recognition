"""Train and evaluate a KNN classifier for Devanagari handwritten digits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import load_digit_images

DIGIT_GLYPHS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size < 8:
        raise ValueError("--image-size should be at least 8 pixels.")
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        raise ValueError("Test and validation sizes must be between 0 and 1.")
    if args.test_size + args.validation_size >= 1:
        raise ValueError("Test size + validation size must be less than 1.")
    if not args.k_values or any(k < 1 for k in args.k_values):
        raise ValueError("--k-values must contain positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y, data_meta = load_digit_images(args.data_dir, image_size=args.image_size)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    validation_fraction = args.validation_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=validation_fraction,
        random_state=args.random_state, stratify=y_trainval,
    )

    comparisons = []
    for k in sorted(set(args.k_values)):
        if k > len(y_train):
            raise ValueError(f"Invalid K={k}; it exceeds training sample count.")
        model = KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="euclidean", n_jobs=-1
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        comparisons.append({
            "k": int(k),
            "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        })

    comparison_df = pd.DataFrame(comparisons).sort_values(
        ["validation_accuracy", "k"], ascending=[False, True]
    )
    best_k = int(comparison_df.iloc[0]["k"])
    comparison_df.to_csv(args.output_dir / "k_comparison.csv", index=False)

    X_final_train = np.concatenate([X_train, X_val], axis=0)
    y_final_train = np.concatenate([y_train, y_val], axis=0)
    final_model = KNeighborsClassifier(
        n_neighbors=best_k, weights="distance", metric="euclidean", n_jobs=-1
    )
    final_model.fit(X_final_train, y_final_train)
    y_pred = final_model.predict(X_test)

    labels = list(range(10))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_text = classification_report(
        y_test, y_pred, labels=labels, target_names=DIGIT_GLYPHS,
        digits=4, zero_division=0,
    )
    (args.output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    pd.DataFrame(cm, index=DIGIT_GLYPHS, columns=DIGIT_GLYPHS).to_csv(
        args.output_dir / "confusion_matrix.csv", index=True
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[str(i) for i in labels]
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"KNN Confusion Matrix (K={best_k})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    metrics = {
        "project": "Devanagari Handwritten Digit Recognition Using KNN",
        "dataset_directory": data_meta["digits_directory"],
        "total_images_loaded": data_meta["image_count"],
        "class_counts": data_meta["class_counts"],
        "image_size": args.image_size,
        "features_per_image": data_meta["feature_count"],
        "split_counts": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "final_training_after_refit": int(len(y_final_train)),
            "test": int(len(y_test)),
        },
        "candidate_k_validation_accuracy": comparisons,
        "selected_k": best_k,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_precision": float(precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )),
        "test_macro_recall": float(recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )),
        "test_macro_f1": float(f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )),
        "random_state": args.random_state,
        "note": "Metrics are generated from the local dataset run; they are not pre-filled estimates.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    count = min(100, len(y_test))
    prediction_df = pd.DataFrame({
        "actual_label": y_test[:count],
        "predicted_label": y_pred[:count],
        "actual_glyph": [DIGIT_GLYPHS[int(v)] for v in y_test[:count]],
        "predicted_glyph": [DIGIT_GLYPHS[int(v)] for v in y_pred[:count]],
    })
    prediction_df.to_csv(args.output_dir / "sample_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(report_text)
    print(f"\nSaved evaluation outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
