"""Load and preprocess the digit-only portion of the Devanagari handwritten dataset."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VALID_PREPROCESS_MODES = {"raw", "centered"}

DIGIT_GLYPHS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]

LABEL_ALIASES = {
    "zero": 0, "shunya": 0, "sunya": 0, "०": 0,
    "one": 1, "ek": 1, "१": 1,
    "two": 2, "do": 2, "२": 2,
    "three": 3, "teen": 3, "तीन": 3, "३": 3,
    "four": 4, "char": 4, "चार": 4, "४": 4,
    "five": 5, "panch": 5, "पाँच": 5, "५": 5,
    "six": 6, "chhah": 6, "छह": 6, "६": 6,
    "seven": 7, "saat": 7, "सात": 7, "७": 7,
    "eight": 8, "aath": 8, "आठ": 8, "८": 8,
    "nine": 9, "nau": 9, "नौ": 9, "९": 9,
}


def normalize_folder_label(folder_name: str) -> int | None:
    """Return an integer digit label (0-9) when a folder or file name is recognizable."""
    name = folder_name.strip().lower()
    if name in LABEL_ALIASES:
        return LABEL_ALIASES[name]
    if len(name) == 1:
        try:
            value = unicodedata.digit(name)
            if 0 <= value <= 9:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?<!\d)([0-9])$", name)
    return int(match.group(1)) if match else None


def find_digits_directory(data_dir: Path) -> Path:
    """Find a directory named DIGITS, case-insensitively."""
    data_dir = Path(data_dir)
    if data_dir.is_dir() and data_dir.name.lower() == "digits":
        return data_dir
    candidates = [
        path for path in data_dir.rglob("*")
        if path.is_dir() and path.name.lower() == "digits"
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a DIGITS directory below {data_dir}. "
            "Extract the Kaggle archive into data/raw/ and check its folder structure."
        )
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def _count_csv_lines(csv_path: Path) -> int:
    """Count lines in a CSV file quickly in binary mode."""
    with csv_path.open("rb") as handle:
        return sum(1 for _ in handle)


def _inspect_excluded_char_directory(digits_dir: Path) -> dict[str, Any]:
    """Summarize excluded alphabet files under sibling CHAR/ directory if present."""
    parent = digits_dir.parent
    char_candidates = [
        p for p in parent.iterdir()
        if p.is_dir() and p.name.lower() == "char"
    ] if parent.is_dir() else []

    if not char_candidates:
        return {"present": False, "directory": None, "csv_files": 0, "rows": 0, "image_files": 0}

    char_dir = char_candidates[0]
    csv_dir = next((p for p in char_dir.iterdir() if p.is_dir() and p.name.lower() == "csv_files"), None)
    if csv_dir is not None:
        csv_files = sorted(csv_dir.glob("*.csv"))
        rows = sum(_count_csv_lines(p) for p in csv_files)
        return {
            "present": True,
            "directory": str(char_dir),
            "source_subdir": str(csv_dir),
            "csv_files": len(csv_files),
            "rows": int(rows),
            "image_files": 0,
        }

    image_files = sum(
        1 for p in char_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {
        "present": True,
        "directory": str(char_dir),
        "source_subdir": None,
        "csv_files": 0,
        "rows": 0,
        "image_files": int(image_files),
    }


def preprocess_image_array(
    image_array: np.ndarray,
    image_size: int = 16,
    preprocess_mode: str = "raw",
    border_crop: int = 3,
    ink_threshold: int = 200,
) -> np.ndarray:
    """Convert a 2D grayscale uint8 image to a flattened float32 vector in [0, 1]."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8 pixels.")
    if preprocess_mode not in VALID_PREPROCESS_MODES:
        raise ValueError(
            f"Unsupported preprocess_mode '{preprocess_mode}'. "
            f"Expected one of {sorted(VALID_PREPROCESS_MODES)}."
        )

    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image_array.shape}.")

    arr_u8 = np.clip(image_array, 0, 255).astype(np.uint8)

    if preprocess_mode == "raw":
        pil_img = Image.fromarray(arr_u8, mode="L").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        return (np.asarray(pil_img, dtype=np.float32) / 255.0).reshape(-1)

    # Centered mode: clean canvas border artifacts, crop tight ink bounding box,
    # pad symmetrically to square to preserve aspect ratio, and resize with margin.
    work = arr_u8.copy()
    h_orig, w_orig = work.shape
    b = min(border_crop, max(0, min(h_orig, w_orig) // 6))
    if b > 0:
        work[:b, :] = 255
        work[-b:, :] = 255
        work[:, :b] = 255
        work[:, -b:] = 255

    mask = work < ink_threshold
    if not np.any(mask):
        pil_img = Image.fromarray(arr_u8, mode="L").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        return (np.asarray(pil_img, dtype=np.float32) / 255.0).reshape(-1)

    ys, xs = np.where(mask)
    crop = work[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    side = max(ch, cw)
    square = np.full((side, side), 255, dtype=np.uint8)
    pad_top = (side - ch) // 2
    pad_left = (side - cw) // 2
    square[pad_top:pad_top + ch, pad_left:pad_left + cw] = crop

    inner_size = max(4, image_size - 2)
    resized_crop = Image.fromarray(square, mode="L").resize(
        (inner_size, inner_size), Image.Resampling.LANCZOS
    )
    canvas = Image.new("L", (image_size, image_size), color=255)
    offset = (image_size - inner_size) // 2
    canvas.paste(resized_crop, (offset, offset))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).reshape(-1)


def preprocess_images(
    raw_images: list[np.ndarray],
    image_size: int = 16,
    preprocess_mode: str = "raw",
) -> np.ndarray:
    """Preprocess a list of 2D uint8 grayscale images into a 2D float32 feature matrix."""
    if not raw_images:
        raise ValueError("raw_images must not be empty.")
    vectors = [
        preprocess_image_array(img, image_size=image_size, preprocess_mode=preprocess_mode)
        for img in raw_images
    ]
    return np.asarray(vectors, dtype=np.float32)


def _load_from_csv_directory(
    digits_dir: Path,
    csv_dir: Path,
) -> tuple[list[np.ndarray], list[int], list[str], dict[str, Any]]:
    """Load 65x65 flattened grayscale rows from DIGITS/CSV_FILES/*.csv."""
    raw_images: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    ignored_files: list[str] = []
    corrupt_files: list[dict[str, str]] = []
    corrupt_rows = 0
    detected_dimensions: set[str] = set()

    for csv_path in sorted(csv_dir.glob("*.csv"), key=lambda p: (len(p.stem), p.stem)):
        label = normalize_folder_label(csv_path.stem)
        if label is None:
            ignored_files.append(csv_path.name)
            continue

        try:
            df = pd.read_csv(csv_path, header=None)
        except Exception as exc:  # pragma: no cover - defensive file error handling
            corrupt_files.append({"file": str(csv_path), "error": str(exc)})
            continue

        if df.empty:
            corrupt_files.append({"file": str(csv_path), "error": "Empty CSV file"})
            continue

        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        valid_mask = ~numeric_df.isna().any(axis=1)
        if not bool(valid_mask.all()):
            corrupt_rows += int((~valid_mask).sum())
            numeric_df = numeric_df.loc[valid_mask]

        values = numeric_df.to_numpy(dtype=np.float64)
        n_cols = values.shape[1]
        side = math.isqrt(n_cols)
        if side * side != n_cols or side < 8:
            corrupt_files.append({
                "file": str(csv_path),
                "error": f"Expected square pixel row length, got {n_cols} columns",
            })
            continue

        in_range_mask = (values >= 0.0).all(axis=1) & (values <= 255.0).all(axis=1)
        if not bool(in_range_mask.all()):
            corrupt_rows += int((~in_range_mask).sum())
            values = values[in_range_mask]

        detected_dimensions.add(f"{side}x{side}")
        arr_u8 = np.rint(values).astype(np.uint8).reshape(-1, side, side)
        for row_idx in range(len(arr_u8)):
            raw_images.append(arr_u8[row_idx])
            labels.append(label)
            sample_ids.append(f"{csv_path.name}#row_{row_idx}")

    info = {
        "source_format": "csv",
        "source_directory": str(csv_dir),
        "detected_dimensions": sorted(detected_dimensions),
        "ignored_files": sorted(ignored_files),
        "corrupt_files": corrupt_files,
        "corrupt_rows": int(corrupt_rows),
    }
    return raw_images, labels, sample_ids, info


def _load_from_image_directory(
    digits_dir: Path,
) -> tuple[list[np.ndarray], list[int], list[str], dict[str, Any]]:
    """Load grayscale images from DIGITS/<class-folder>/*.*."""
    raw_images: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    ignored_folders: set[str] = set()
    corrupt_files: list[dict[str, str]] = []
    detected_dimensions: set[str] = set()
    detected_formats: set[str] = set()

    for image_path in sorted(digits_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(digits_dir)
        if len(relative.parts) < 2:
            continue
        class_folder = relative.parts[0]
        label = normalize_folder_label(class_folder)
        if label is None:
            ignored_folders.add(class_folder)
            continue

        try:
            with Image.open(image_path) as image:
                detected_formats.add((image.format or image_path.suffix.lstrip(".")).upper())
                detected_dimensions.add(f"{image.width}x{image.height}")
                gray = image.convert("L")
                arr_u8 = np.asarray(gray, dtype=np.uint8)
            raw_images.append(arr_u8)
            labels.append(label)
            sample_ids.append(str(image_path))
        except (OSError, ValueError) as exc:
            corrupt_files.append({"file": str(image_path), "error": str(exc)})

    info = {
        "source_format": "image_folders",
        "source_directory": str(digits_dir),
        "detected_formats": sorted(detected_formats),
        "detected_dimensions": sorted(detected_dimensions),
        "ignored_files": sorted(ignored_folders),
        "corrupt_files": corrupt_files,
        "corrupt_rows": 0,
    }
    return raw_images, labels, sample_ids, info


def _deduplicate_samples(
    raw_images: list[np.ndarray],
    labels: list[int],
    sample_ids: list[str],
) -> tuple[list[np.ndarray], list[int], list[str], int, int]:
    """Remove exact duplicate image matrices to prevent train/val/test data leakage."""
    seen_hashes: dict[str, int] = {}
    unique_images: list[np.ndarray] = []
    unique_labels: list[int] = []
    unique_ids: list[str] = []
    duplicates_removed = 0
    cross_class_duplicates = 0

    for img, label, sid in zip(raw_images, labels, sample_ids):
        digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
        if digest in seen_hashes:
            duplicates_removed += 1
            if seen_hashes[digest] != label:
                cross_class_duplicates += 1
            continue
        seen_hashes[digest] = label
        unique_images.append(img)
        unique_labels.append(label)
        unique_ids.append(sid)

    return unique_images, unique_labels, unique_ids, duplicates_removed, cross_class_duplicates


def load_raw_digit_dataset(
    data_dir: Path,
    deduplicate: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Load raw grayscale digit arrays (0-9) and full inspection metadata."""
    digits_dir = find_digits_directory(Path(data_dir))
    csv_dir = next(
        (p for p in digits_dir.iterdir() if p.is_dir() and p.name.lower() == "csv_files"),
        None,
    )

    if csv_dir is not None and any(csv_dir.glob("*.csv")):
        raw_images, labels, sample_ids, source_info = _load_from_csv_directory(digits_dir, csv_dir)
    else:
        raw_images, labels, sample_ids, source_info = _load_from_image_directory(digits_dir)

    if not raw_images:
        raise RuntimeError(
            f"No valid digit images were loaded from {digits_dir}. "
            "Expected CSV files under DIGITS/CSV_FILES/ or image files under DIGITS/<class>/."
        )

    y_raw = np.asarray(labels, dtype=np.int64)
    raw_unique, raw_counts = np.unique(y_raw, return_counts=True)
    raw_class_counts = {str(int(lbl)): int(cnt) for lbl, cnt in zip(raw_unique, raw_counts)}

    if deduplicate:
        (
            kept_images,
            kept_labels,
            kept_ids,
            duplicates_removed,
            cross_class_duplicates,
        ) = _deduplicate_samples(raw_images, labels, sample_ids)
    else:
        kept_images, kept_labels, kept_ids = raw_images, labels, sample_ids
        duplicates_removed = 0
        cross_class_duplicates = 0

    y = np.asarray(kept_labels, dtype=np.int64)
    present = sorted(np.unique(y).tolist())
    if present != list(range(10)):
        raise RuntimeError(
            f"Expected all ten digit labels 0-9, but detected {present}. "
            f"Ignored items: {source_info['ignored_files']}."
        )

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {str(int(lbl)): int(cnt) for lbl, cnt in zip(unique, counts)}
    char_info = _inspect_excluded_char_directory(digits_dir)

    metadata: dict[str, Any] = {
        "digits_directory": str(digits_dir),
        "source_format": source_info["source_format"],
        "source_directory": source_info["source_directory"],
        "detected_dimensions": source_info["detected_dimensions"],
        "detected_formats": source_info.get("detected_formats", ["CSV (uint8 pixel matrix)"]),
        "raw_image_count": int(len(raw_images)),
        "raw_class_counts": raw_class_counts,
        "deduplicated": bool(deduplicate),
        "duplicates_removed": int(duplicates_removed),
        "cross_class_duplicates": int(cross_class_duplicates),
        "image_count": int(len(y)),
        "class_counts": class_counts,
        "ignored_class_folders": source_info["ignored_files"],
        "corrupt_files": source_info["corrupt_files"],
        "corrupt_rows_skipped": int(source_info["corrupt_rows"]),
        "excluded_alphabet_subset": char_info,
        "sample_paths": kept_ids,
    }
    return kept_images, y, metadata


def load_digit_images(
    data_dir: Path,
    image_size: int = 16,
    preprocess_mode: str = "raw",
    deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load digits from DIGITS/ and return flattened normalized pixel vectors."""
    raw_images, y, metadata = load_raw_digit_dataset(data_dir, deduplicate=deduplicate)
    X = preprocess_images(raw_images, image_size=image_size, preprocess_mode=preprocess_mode)
    metadata["image_size"] = int(image_size)
    metadata["preprocess_mode"] = preprocess_mode
    metadata["feature_count"] = int(X.shape[1])
    return X, y, metadata
