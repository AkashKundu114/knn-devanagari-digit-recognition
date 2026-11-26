"""Load the digit-only portion of the Devanagari handwritten dataset."""
from pathlib import Path
import re
import unicodedata

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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


def normalize_folder_label(folder_name: str):
    """Return an integer digit label when the folder name is recognizable."""
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


def load_digit_images(data_dir: Path, image_size: int = 16):
    """Load images from DIGITS/<class-folder>; return flattened normalized pixels."""
    digits_dir = find_digits_directory(Path(data_dir))
    images, labels, paths = [], [], []
    ignored_folders = set()

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
                image = image.convert("L").resize(
                    (image_size, image_size), Image.Resampling.LANCZOS
                )
                array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(array.reshape(-1))
            labels.append(label)
            paths.append(str(image_path))
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable image {image_path}: {exc}")

    if not images:
        raise RuntimeError(
            f"No digit images were loaded from {digits_dir}. "
            "Expected image files inside ten class folders under DIGITS/."
        )

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    present = sorted(np.unique(y).tolist())
    if present != list(range(10)):
        raise RuntimeError(
            f"Expected all ten digit labels 0-9, but detected {present}. "
            f"Unrecognized class folders: {sorted(ignored_folders)}. "
            "Inspect DIGITS and update LABEL_ALIASES if needed."
        )

    unique, counts = np.unique(y, return_counts=True)
    metadata = {
        "digits_directory": str(digits_dir),
        "image_count": int(len(y)),
        "class_counts": {str(int(label)): int(count) for label, count in zip(unique, counts)},
        "image_size": int(image_size),
        "feature_count": int(X.shape[1]),
        "ignored_class_folders": sorted(ignored_folders),
        "sample_paths": paths,
    }
    return X, y, metadata
