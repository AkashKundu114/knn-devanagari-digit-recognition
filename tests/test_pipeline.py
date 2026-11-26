"""Unit and integration tests for the Devanagari KNN digit recognition pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data_loader import (
    find_digits_directory,
    load_digit_images,
    load_raw_digit_dataset,
    normalize_folder_label,
    preprocess_image_array,
)
from src.train_evaluate import (
    create_stratified_splits,
    run_experiment,
    select_best_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"


def _make_synthetic_digit_array(digit: int, variant: int, size: int = 65) -> np.ndarray:
    """Create a deterministic synthetic grayscale 65x65 array per digit class."""
    arr = np.full((size, size), 255, dtype=np.uint8)
    row_start = 10 + (digit % 5) * 9 + (variant % 4)
    col_start = 10 + (digit // 5) * 22 + ((variant // 4) % 4)
    arr[row_start:row_start + 8, col_start:col_start + 12] = 0
    return arr


def _build_synthetic_csv_dataset(root: Path, per_class: int = 10, include_dups: bool = True) -> Path:
    """Create a synthetic DEVA-C-SET-CSV-like dataset under root."""
    digits_csv_dir = root / "DEVA-C-SET-CSV" / "DIGITS" / "CSV_FILES"
    char_csv_dir = root / "DEVA-C-SET-CSV" / "CHAR" / "CSV_FILES"
    digits_csv_dir.mkdir(parents=True, exist_ok=True)
    char_csv_dir.mkdir(parents=True, exist_ok=True)

    for digit in range(10):
        rows = [_make_synthetic_digit_array(digit, v).reshape(-1) for v in range(per_class)]
        if include_dups:
            rows.append(rows[0].copy())
        pd.DataFrame(rows).to_csv(digits_csv_dir / f"{digit}.csv", header=False, index=False)

    pd.DataFrame([np.zeros(4225, dtype=np.uint8)]).to_csv(
        char_csv_dir / "1.csv", header=False, index=False
    )
    return root


def _build_synthetic_image_dataset(root: Path, per_class: int = 8) -> Path:
    """Create a synthetic image-folder dataset under root/DIGITS/<class>/."""
    digits_dir = root / "DIGITS"
    for digit in range(10):
        cls_dir = digits_dir / f"digit_{digit}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for v in range(per_class):
            arr = _make_synthetic_digit_array(digit, v)
            Image.fromarray(arr, mode="L").convert("RGB").save(cls_dir / f"sample_{v}.png")
    return root


@pytest.mark.parametrize(
    "folder_name,expected",
    [
        ("0", 0),
        ("9", 9),
        ("०", 0),
        ("५", 5),
        ("nine", 9),
        ("shunya", 0),
        ("digit_3", 3),
        ("class-7", 7),
        ("char_ka", None),
        ("unknown", None),
    ],
)
def test_normalize_folder_label(folder_name: str, expected: int | None) -> None:
    assert normalize_folder_label(folder_name) == expected


def test_find_digits_directory_and_invalid_path(tmp_path: Path) -> None:
    nested = tmp_path / "archive" / "Digits"
    nested.mkdir(parents=True)
    assert find_digits_directory(tmp_path) == nested

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        find_digits_directory(empty_dir)

    with pytest.raises(FileNotFoundError):
        find_digits_directory(tmp_path / "nonexistent")


def test_missing_classes_raises_error(tmp_path: Path) -> None:
    digits_csv_dir = tmp_path / "DIGITS" / "CSV_FILES"
    digits_csv_dir.mkdir(parents=True)
    for digit in range(9):
        rows = [_make_synthetic_digit_array(digit, v).reshape(-1) for v in range(4)]
        pd.DataFrame(rows).to_csv(digits_csv_dir / f"{digit}.csv", header=False, index=False)

    with pytest.raises(RuntimeError, match="Expected all ten digit labels"):
        load_digit_images(tmp_path, image_size=16)


def test_preprocessing_resizing_and_normalization() -> None:
    sample = _make_synthetic_digit_array(4, 1, size=65)
    for mode in ("raw", "centered"):
        for size in (16, 32):
            vec = preprocess_image_array(sample, image_size=size, preprocess_mode=mode)
            assert vec.dtype == np.float32
            assert vec.shape == (size * size,)
            assert float(vec.min()) >= 0.0
            assert float(vec.max()) <= 1.0
            assert float(vec.min()) < float(vec.max())

    with pytest.raises(ValueError):
        preprocess_image_array(sample, image_size=4, preprocess_mode="raw")
    with pytest.raises(ValueError):
        preprocess_image_array(sample, image_size=16, preprocess_mode="invalid")


def test_csv_loader_class_filtering_dedup_and_corrupt_rows(tmp_path: Path) -> None:
    data_dir = _build_synthetic_csv_dataset(tmp_path, per_class=10, include_dups=True)
    csv_0 = data_dir / "DEVA-C-SET-CSV" / "DIGITS" / "CSV_FILES" / "0.csv"
    with csv_0.open("a", encoding="utf-8") as fh:
        fh.write(",".join(["bad"] * 4225) + "\n")
    (csv_0.parent / "notes.csv").write_text("1,2,3\n", encoding="utf-8")

    X, y, meta = load_digit_images(data_dir, image_size=16, preprocess_mode="centered", deduplicate=True)
    assert meta["source_format"] == "csv"
    assert meta["raw_image_count"] == 110
    assert meta["duplicates_removed"] == 10
    assert meta["image_count"] == 100
    assert meta["corrupt_rows_skipped"] == 1
    assert meta["ignored_class_folders"] == ["notes.csv"]
    assert meta["excluded_alphabet_subset"]["present"] is True
    assert meta["excluded_alphabet_subset"]["rows"] == 1
    assert X.shape == (100, 256)
    assert set(np.unique(y).tolist()) == set(range(10))


def test_image_folder_loader_and_corrupt_image_handling(tmp_path: Path) -> None:
    data_dir = _build_synthetic_image_dataset(tmp_path, per_class=8)
    (data_dir / "DIGITS" / "digit_0" / "broken.jpg").write_bytes(b"not-a-valid-jpeg")

    X, y, meta = load_digit_images(data_dir, image_size=16, preprocess_mode="raw", deduplicate=False)
    assert meta["source_format"] == "image_folders"
    assert len(meta["corrupt_files"]) == 1
    assert X.shape == (80, 256)
    assert y.shape == (80,)
    assert float(X.min()) >= 0.0 and float(X.max()) <= 1.0


def test_stratified_splitting_no_leakage() -> None:
    y = np.repeat(np.arange(10), 20)
    idx_train, idx_val, idx_test = create_stratified_splits(
        y, test_size=0.15, validation_size=0.15, random_state=42
    )
    assert len(idx_train) + len(idx_val) + len(idx_test) == 200
    assert len(idx_test) == 30
    assert 29 <= len(idx_val) <= 31
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0
    _, test_counts = np.unique(y[idx_test], return_counts=True)
    assert np.all(test_counts == 3)


def test_knn_configuration_weighting_and_reproducibility(tmp_path: Path) -> None:
    data_dir = _build_synthetic_csv_dataset(tmp_path / "data", per_class=12, include_dups=False)
    raw_imgs, y, _ = load_raw_digit_dataset(data_dir, deduplicate=True)
    idx_train, idx_val, _ = create_stratified_splits(y, random_state=42)
    cache = {
        ("centered", 16): np.asarray([preprocess_image_array(im, 16, "centered") for im in raw_imgs]),
        ("raw", 16): np.asarray([preprocess_image_array(im, 16, "raw") for im in raw_imgs]),
    }
    comp_df, best = select_best_configuration(
        cache, y, idx_train, idx_val, [1, 3, 5], weights_list=["distance", "uniform"]
    )
    assert len(comp_df) == 12
    assert set(comp_df["weights"].unique()) == {"distance", "uniform"}
    assert comp_df.iloc[0]["validation_accuracy"] == best["validation_accuracy"]
    assert best["k"] in (1, 3, 5)
    assert best["weights"] in ("distance", "uniform")

    out_1 = tmp_path / "run1"
    out_2 = tmp_path / "run2"
    m1 = run_experiment(
        data_dir, out_1, image_sizes=[16], preprocess_modes=["centered"],
        k_values=[1, 3, 5], weights_list=["distance", "uniform"], random_state=42,
    )
    m2 = run_experiment(
        data_dir, out_2, image_sizes=[16], preprocess_modes=["centered"],
        k_values=[1, 3, 5], weights_list=["distance", "uniform"], random_state=42,
    )
    assert m1["selected_k"] == m2["selected_k"]
    assert m1["weights"] == m2["weights"]
    assert m1["test_accuracy"] == pytest.approx(m2["test_accuracy"])
    assert (out_1 / "sample_predictions.csv").read_text(encoding="utf-8") == (
        out_2 / "sample_predictions.csv"
    ).read_text(encoding="utf-8")


def test_end_to_end_synthetic_outputs_and_agreement(tmp_path: Path) -> None:
    data_dir = _build_synthetic_csv_dataset(tmp_path / "data", per_class=12, include_dups=True)
    out_dir = tmp_path / "results"
    metrics = run_experiment(
        data_dir=data_dir,
        output_dir=out_dir,
        image_sizes=[16, 32],
        preprocess_modes=["raw", "centered"],
        k_values=[1, 3, 5],
        weights_list=["distance", "uniform"],
        random_state=42,
    )

    expected_files = [
        "metrics.json",
        "dataset_inspection.json",
        "classification_report.txt",
        "classification_report.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "k_comparison.csv",
        "k_validation_curve.png",
        "class_distribution.png",
        "dataset_samples.png",
        "prediction_samples.png",
        "sample_predictions.csv",
    ]
    for name in expected_files:
        assert (out_dir / name).is_file(), f"Missing output artifact: {name}"

    cm = pd.read_csv(out_dir / "confusion_matrix.csv", index_col=0).to_numpy()
    assert cm.shape == (10, 10)
    assert int(cm.sum()) == metrics["split_counts"]["test"]
    assert float(np.trace(cm) / cm.sum()) == pytest.approx(metrics["test_accuracy"])

    cr_df = pd.read_csv(out_dir / "classification_report.csv")
    assert len(cr_df) == 10
    assert int(cr_df["support"].sum()) == metrics["split_counts"]["test"]

    preds = pd.read_csv(out_dir / "sample_predictions.csv")
    assert len(preds) == metrics["split_counts"]["test"]
    assert float(preds["correct"].mean()) == pytest.approx(metrics["test_accuracy"])


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "DEVA-C-SET-CSV" / "DIGITS").is_dir()
    or not (RESULTS_DIR / "metrics.json").is_file(),
    reason="Real Kaggle dataset or generated results not present locally.",
)
def test_real_dataset_results_consistency() -> None:
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    cm = pd.read_csv(RESULTS_DIR / "confusion_matrix.csv", index_col=0).to_numpy()
    k_comp = pd.read_csv(RESULTS_DIR / "k_comparison.csv")
    preds = pd.read_csv(RESULTS_DIR / "sample_predictions.csv")
    cr_df = pd.read_csv(RESULTS_DIR / "classification_report.csv")

    assert metrics["total_images_loaded"] == 1073
    assert metrics["raw_digit_images"] == 8676
    assert metrics["duplicates_removed"] == 7603
    assert len(k_comp) == 40
    assert sum(metrics["class_counts"].values()) == 1073
    assert int(cm.sum()) == metrics["split_counts"]["test"]
    assert float(np.trace(cm) / cm.sum()) == pytest.approx(metrics["test_accuracy"])
    assert float(preds["correct"].mean()) == pytest.approx(metrics["test_accuracy"])
    assert int(cr_df["support"].sum()) == metrics["split_counts"]["test"]
    assert int(k_comp.iloc[0]["k"]) == metrics["selected_k"]
    assert str(k_comp.iloc[0]["weights"]) == metrics["weights"]
    assert str(k_comp.iloc[0]["preprocess_mode"]) == metrics["selected_preprocess_mode"]
    assert int(k_comp.iloc[0]["image_size"]) == metrics["image_size"]
    assert float(k_comp.iloc[0]["validation_accuracy"]) == pytest.approx(
        metrics["selected_validation_accuracy"]
    )
    support_sum = sum(v["support"] for v in metrics["per_class_metrics"].values())
    assert support_sum == metrics["split_counts"]["test"]
