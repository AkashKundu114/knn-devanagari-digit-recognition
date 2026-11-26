"""Load, deduplicate, and preprocess the digit-only Devanagari dataset."""
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
    "zero": 0, "shunya": 0, "sunya": 0, "०": 0, "one": 1, "ek": 1, "१": 1,
    "two": 2, "do": 2, "२": 2, "three": 3, "teen": 3, "तीन": 3, "३": 3,
    "four": 4, "char": 4, "चार": 4, "४": 4, "five": 5, "panch": 5, "पाँच": 5, "५": 5,
    "six": 6, "chhah": 6, "छह": 6, "६": 6, "seven": 7, "saat": 7, "सात": 7, "७": 7,
    "eight": 8, "aath": 8, "आठ": 8, "८": 8, "nine": 9, "nau": 9, "नौ": 9, "९": 9,
}


def normalize_folder_label(folder_name: str) -> int | None:
    """Map a folder or CSV stem to an integer digit label (0-9)."""
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
    """Locate a directory named DIGITS, case-insensitively."""
    data_dir = Path(data_dir)
    if data_dir.is_dir() and data_dir.name.lower() == "digits":
        return data_dir
    candidates = [p for p in data_dir.rglob("*") if p.is_dir() and p.name.lower() == "digits"]
    if not candidates:
        raise FileNotFoundError(f"Could not find a DIGITS directory below {data_dir}.")
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def _inspect_excluded_char(digits_dir: Path) -> dict[str, Any]:
    """Summarize excluded alphabet files under sibling CHAR/ if present."""
    parent = digits_dir.parent
    char_dir = next((p for p in parent.iterdir() if p.is_dir() and p.name.lower() == "char"), None) if parent.is_dir() else None
    if char_dir is None:
        return {"present": False, "directory": None, "csv_files": 0, "rows": 0, "image_files": 0}
    csv_dir = next((p for p in char_dir.iterdir() if p.is_dir() and p.name.lower() == "csv_files"), None)
    if csv_dir is not None:
        csv_files = sorted(csv_dir.glob("*.csv"))
        rows = sum(sum(1 for _ in p.open("rb")) for p in csv_files)
        return {"present": True, "directory": str(char_dir), "source_subdir": str(csv_dir), "csv_files": len(csv_files), "rows": int(rows), "image_files": 0}
    img_count = sum(1 for p in char_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return {"present": True, "directory": str(char_dir), "source_subdir": None, "csv_files": 0, "rows": 0, "image_files": int(img_count)}


def preprocess_image_array(
    image_array: np.ndarray,
    image_size: int = 16,
    preprocess_mode: str = "raw",
    border_crop: int = 3,
    ink_threshold: int = 200,
) -> np.ndarray:
    """Convert a 2D grayscale image to a flattened float32 vector in [0, 1]."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8 pixels.")
    if preprocess_mode not in VALID_PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocess_mode '{preprocess_mode}'.")
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image_array.shape}.")

    arr_u8 = np.clip(image_array, 0, 255).astype(np.uint8)
    if preprocess_mode == "raw":
        resized = Image.fromarray(arr_u8, mode="L").resize((image_size, image_size), Image.Resampling.LANCZOS)
        return (np.asarray(resized, dtype=np.float32) / 255.0).reshape(-1)

    work = arr_u8.copy()
    b = min(border_crop, max(0, min(work.shape) // 6))
    if b > 0:
        work[:b, :] = work[-b:, :] = work[:, :b] = work[:, -b:] = 255
    mask = work < ink_threshold
    if not np.any(mask):
        resized = Image.fromarray(arr_u8, mode="L").resize((image_size, image_size), Image.Resampling.LANCZOS)
        return (np.asarray(resized, dtype=np.float32) / 255.0).reshape(-1)

    ys, xs = np.where(mask)
    crop = work[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = max(crop.shape)
    square = np.full((side, side), 255, dtype=np.uint8)
    top, left = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    square[top:top + crop.shape[0], left:left + crop.shape[1]] = crop

    inner = max(4, image_size - 2)
    pil_crop = Image.fromarray(square, mode="L").resize((inner, inner), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (image_size, image_size), color=255)
    canvas.paste(pil_crop, ((image_size - inner) // 2, (image_size - inner) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).reshape(-1)


def preprocess_images(raw_images: list[np.ndarray], image_size: int = 16, preprocess_mode: str = "raw") -> np.ndarray:
    """Preprocess a list of 2D uint8 grayscale images into a float32 feature matrix."""
    if not raw_images:
        raise ValueError("raw_images must not be empty.")
    return np.asarray([preprocess_image_array(img, image_size, preprocess_mode) for img in raw_images], dtype=np.float32)


def _load_csv_digits(csv_dir: Path) -> tuple[list[np.ndarray], list[int], list[str], dict[str, Any]]:
    """Load flattened square grayscale matrices from DIGITS/CSV_FILES/*.csv."""
    images, labels, ids, ignored, corrupt_files, dims, corrupt_rows = [], [], [], [], [], set(), 0
    for csv_path in sorted(csv_dir.glob("*.csv"), key=lambda p: (len(p.stem), p.stem)):
        label = normalize_folder_label(csv_path.stem)
        if label is None:
            ignored.append(csv_path.name)
            continue
        try:
            num_df = pd.read_csv(csv_path, header=None).apply(pd.to_numeric, errors="coerce")
        except Exception as exc:  # pragma: no cover
            corrupt_files.append({"file": str(csv_path), "error": str(exc)})
            continue

        valid = ~num_df.isna().any(axis=1)
        corrupt_rows += int((~valid).sum())
        vals = num_df.loc[valid].to_numpy(dtype=np.float64)
        side = math.isqrt(vals.shape[1]) if vals.size else 0
        if side * side != vals.shape[1] or side < 8:
            corrupt_files.append({"file": str(csv_path), "error": f"Invalid column count {vals.shape[1]}"})
            continue

        in_range = (vals >= 0.0).all(axis=1) & (vals <= 255.0).all(axis=1)
        corrupt_rows += int((~in_range).sum())
        dims.add(f"{side}x{side}")
        arr_u8 = np.rint(vals[in_range]).astype(np.uint8).reshape(-1, side, side)
        images.extend(arr_u8)
        labels.extend([label] * len(arr_u8))
        ids.extend(f"{csv_path.name}#row_{i}" for i in range(len(arr_u8)))

    return images, labels, ids, {
        "source_format": "csv", "source_directory": str(csv_dir), "detected_dimensions": sorted(dims),
        "detected_formats": ["CSV (uint8 pixel matrix)"], "ignored_files": sorted(ignored),
        "corrupt_files": corrupt_files, "corrupt_rows": int(corrupt_rows),
    }


def _load_image_folders(digits_dir: Path) -> tuple[list[np.ndarray], list[int], list[str], dict[str, Any]]:
    """Load grayscale images from DIGITS/<class-folder>/*.*."""
    images, labels, ids, ignored, corrupt_files, dims, fmts = [], [], [], set(), [], set(), set()
    for path in sorted(digits_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(digits_dir)
        if len(rel.parts) < 2:
            continue
        label = normalize_folder_label(rel.parts[0])
        if label is None:
            ignored.add(rel.parts[0])
            continue
        try:
            with Image.open(path) as img:
                fmts.add((img.format or path.suffix.lstrip(".")).upper())
                dims.add(f"{img.width}x{img.height}")
                images.append(np.asarray(img.convert("L"), dtype=np.uint8))
            labels.append(label)
            ids.append(str(path))
        except (OSError, ValueError) as exc:
            corrupt_files.append({"file": str(path), "error": str(exc)})

    return images, labels, ids, {
        "source_format": "image_folders", "source_directory": str(digits_dir),
        "detected_dimensions": sorted(dims), "detected_formats": sorted(fmts),
        "ignored_files": sorted(ignored), "corrupt_files": corrupt_files, "corrupt_rows": 0,
    }


def load_raw_digit_dataset(data_dir: Path, deduplicate: bool = True) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Load raw grayscale digit arrays (0-9), deduplicate, and return inspection metadata."""
    digits_dir = find_digits_directory(Path(data_dir))
    csv_dir = next((p for p in digits_dir.iterdir() if p.is_dir() and p.name.lower() == "csv_files"), None)
    raw_images, labels, sample_ids, info = _load_csv_digits(csv_dir) if (csv_dir and any(csv_dir.glob("*.csv"))) else _load_image_folders(digits_dir)
    if not raw_images:
        raise RuntimeError(f"No valid digit images loaded from {digits_dir}.")

    u_raw, c_raw = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    raw_counts = {str(int(k)): int(v) for k, v in zip(u_raw, c_raw)}

    seen: dict[str, int] = {}
    kept_imgs, kept_lbls, kept_ids, dups_removed, cross_dups = [], [], [], 0, 0
    for img, lbl, sid in zip(raw_images, labels, sample_ids):
        if deduplicate:
            digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
            if digest in seen:
                dups_removed += 1
                cross_dups += int(seen[digest] != lbl)
                continue
            seen[digest] = lbl
        kept_imgs.append(img)
        kept_lbls.append(lbl)
        kept_ids.append(sid)

    y = np.asarray(kept_lbls, dtype=np.int64)
    if sorted(np.unique(y).tolist()) != list(range(10)):
        raise RuntimeError(f"Expected all ten digit labels 0-9, got {sorted(np.unique(y).tolist())}.")

    u_kept, c_kept = np.unique(y, return_counts=True)
    return kept_imgs, y, {
        "digits_directory": str(digits_dir), "source_format": info["source_format"],
        "source_directory": info["source_directory"], "detected_dimensions": info["detected_dimensions"],
        "detected_formats": info["detected_formats"], "raw_image_count": int(len(raw_images)),
        "raw_class_counts": raw_counts, "deduplicated": bool(deduplicate),
        "duplicates_removed": int(dups_removed), "cross_class_duplicates": int(cross_dups),
        "image_count": int(len(y)), "class_counts": {str(int(k)): int(v) for k, v in zip(u_kept, c_kept)},
        "ignored_class_folders": info["ignored_files"], "corrupt_files": info["corrupt_files"],
        "corrupt_rows_skipped": info["corrupt_rows"], "excluded_alphabet_subset": _inspect_excluded_char(digits_dir),
        "sample_paths": kept_ids,
    }


def load_digit_images(
    data_dir: Path,
    image_size: int = 16,
    preprocess_mode: str = "raw",
    deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load digits from DIGITS/ and return flattened normalized pixel vectors."""
    raw_images, y, metadata = load_raw_digit_dataset(data_dir, deduplicate=deduplicate)
    X = preprocess_images(raw_images, image_size=image_size, preprocess_mode=preprocess_mode)
    metadata.update({"image_size": int(image_size), "preprocess_mode": preprocess_mode, "feature_count": int(X.shape[1])})
    return X, y, metadata
