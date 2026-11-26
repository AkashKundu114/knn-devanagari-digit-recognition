# Devanagari Handwritten Digit Recognition Using KNN

A reproducible classical machine-learning project for classifying handwritten Devanagari numerals (`०` to `९`, classes 0 through 9) with K-Nearest Neighbors (KNN), built for **Machine Learning Assignment 3**.

## Overview

- **Task:** 10-class handwritten digit classification (`०` through `९`).
- **Dataset:** [DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES](https://www.kaggle.com/dsv/9820160) (`sanjaypate/devanagari-script-digital-44000-images-dataset`, DOI: `10.34740/KAGGLE/DSV/9820160`).
- **Actual inspected data format:** The downloaded archive extracts to `DEVA-C-SET-CSV/` and stores flattened 65 x 65 grayscale pixel rows (4,225 integer columns in `[0, 255]`, no header) in CSV files rather than JPEG image folders.
- **Subset and deduplication:** All 35 alphabet files under `CHAR/CSV_FILES/` (`32,855` rows) are excluded. Inside `DIGITS/CSV_FILES/` (`0.csv` to `9.csv`), there are `8,676` raw rows containing `7,603` exact byte-identical duplicates. Removing duplicates yields **1,073 unique digit images** (`0` cross-class duplicates) and prevents train-test data leakage.
- **Preprocessing compared:** `raw` resizing vs. `centered` preprocessing (3-pixel border cleanup, tight bounding-box crop on ink `< 200`, symmetric square padding, Lanczos resize) at `16 x 16` (`256` features) and `32 x 32` (`1,024` features), normalized to `[0, 1]`.
- **Classifier:** Scikit-learn `KNeighborsClassifier` with Euclidean distance, comparing `distance` vs. `uniform` neighbor weighting across `K in {1, 3, 5, 7, 9}`.
- **Data split:** Stratified 70% train (`751`), 15% validation (`161`), and 15% held-out test (`161`) with `random_state = 42`.

## Installation

Python 3.10 or newer is recommended (verified on Python 3.14.6).

```bash
git clone https://github.com/AkashKundu114/knn-devanagari-digit-recognition.git
cd knn-devanagari-digit-recognition
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset setup

1. Configure Kaggle authentication locally (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`). Never commit credentials.
2. Download the dataset using the Kaggle CLI or from [Kaggle](https://www.kaggle.com/dsv/9820160):

```bash
kaggle datasets download -d sanjaypate/devanagari-script-digital-44000-images-dataset
```

3. Extract the archive under `data/raw/` so `data/raw/DEVA-C-SET-CSV/DIGITS/CSV_FILES/0.csv` through `9.csv` exist.
4. Do not commit raw dataset files, archives, or API credentials.

See [DATASET.md](DATASET.md) for the full dataset inspection details.

## Repository structure

```text
.
├── .github/workflows/
│   ├── ci.yml                     # Python CI, pytest, link check, smoke test
│   └── manual-evaluation.yml      # Manual Kaggle evaluation workflow
├── data/
│   └── raw/                       # Local extracted Kaggle dataset (ignored by Git)
├── results/                       # Generated metrics, CSVs, and plots (ignored by Git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # CSV/image loading, deduplication, preprocessing
│   └── train_evaluate.py          # Stratified splits, grid search, evaluation, plots
├── tests/
│   └── test_pipeline.py           # Unit and integration tests
├── .gitignore
├── CHANGELOG.md
├── DATASET.md
├── EVALUATION.md
├── LICENSE
├── METHODOLOGY.md
├── README.md
└── requirements.txt
```

## Execution commands

Run the full KNN experiment (40 candidates: `raw` and `centered` x `16 x 16` and `32 x 32` x `distance` and `uniform` x `K = 1, 3, 5, 7, 9`):

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results
```

Run Python compilation checks and tests:

```bash
python -m compileall src tests
python -m pytest -v
```

## Evaluation methodology and actual results

All 40 candidate configurations are evaluated strictly on the 161-sample validation split. The winning configuration is refitted on combined train + validation (`912` samples) and evaluated once on the untouched `161`-sample test set:

- **Selected configuration:** `centered`, `16 x 16` (256 features), `K = 3`, Euclidean distance, `distance` weighting
- **Validation accuracy (selected):** `99.38%` (`raw` peaked at `69.57%` with `32 x 32`, `uniform`, `K = 3`)
- **Test accuracy:** `95.03%` (`153 / 161` correct)
- **Macro precision:** `95.82%`
- **Macro recall:** `94.29%`
- **Macro F1-score:** `94.86%`
- **Weighted precision / recall / F1:** `95.29%` / `95.03%` / `95.01%`

See [METHODOLOGY.md](METHODOLOGY.md) and [EVALUATION.md](EVALUATION.md) for detailed configuration and per-class observations.

## Reproducibility

- Fixed seed `random_state = 42` for stratified 70 / 15 / 15 splitting.
- Per-image preprocessing fits no global dataset statistics across splits.

## Limitations

- Removing `7,603` duplicate rows leaves `1,073` unique digit images (`59` for digit `०`, `9` in test).
- Uncentered raw pixel Euclidean distance reaches at most `69.57%` validation accuracy due to canvas translation and scale differences.
- Faint toolbar artifacts beyond the 3-pixel border in two test `०` images expanded their bounding boxes and caused `० -> ६` errors.
- KNN inference scales as $O(N \cdot D)$ with training set size, and digital canvas drawings do not guarantee generalization to paper scans or camera photos.

## License and citation

Code is released under the [MIT License](LICENSE). The Kaggle dataset listing specifies CC0 Public Domain.

Dataset citation: Sanjay E. Pate, *DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES*, Kaggle, DOI: `10.34740/KAGGLE/DSV/9820160`.
