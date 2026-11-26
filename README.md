# Devanagari Handwritten Digit Recognition Using KNN

A reproducible classical machine-learning project for classifying handwritten Devanagari numerals (`०` to `९`, classes 0 through 9) with K-Nearest Neighbors (KNN).

## Project overview

- **Task:** 10-class handwritten digit classification (`०` through `९`).
- **Dataset:** [DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES](https://www.kaggle.com/dsv/9820160) (`sanjaypate/devanagari-script-digital-44000-images-dataset`, DOI: `10.34740/KAGGLE/DSV/9820160`).
- **Actual inspected data format:** The downloaded Kaggle archive extracts to `DEVA-C-SET-CSV/` and stores flattened 65 x 65 grayscale pixel matrices (4,225 integer columns in `[0, 255]`, no header) in CSV files rather than JPEG image folders.
- **Subset and deduplication:** All 35 alphabet files under `CHAR/CSV_FILES/` (32,855 rows) are excluded. Inside `DIGITS/CSV_FILES/` (`0.csv` to `9.csv`), there are 8,676 raw rows containing 7,603 exact byte-identical duplicates. Removing duplicates yields **1,073 unique digit images** (0 cross-class duplicates) and prevents train-test data leakage.
- **Preprocessing compared:** `raw` resizing vs. `centered` preprocessing (3-pixel border cleanup, tight ink bounding-box crop, symmetric square padding, Lanczos resize) at `16 x 16` (256 features) and `32 x 32` (1,024 features), normalized to `[0, 1]`.
- **Classifier:** Scikit-learn `KNeighborsClassifier` with Euclidean distance and distance-weighted voting (`K in {1, 3, 5, 7, 9}`).
- **Data split:** Stratified 70% train (751), 15% validation (161), and 15% held-out test (161) with `random_state = 42`.

## Features

- Loads the real Kaggle `DEVA-C-SET-CSV/DIGITS/CSV_FILES/` layout and also supports image-folder datasets (`DIGITS/<class>/*.jpg|png`).
- Detects and skips corrupt rows or unreadable image files safely and logs them in `results/dataset_inspection.json`.
- Removes exact SHA-256 duplicates before splitting so identical images never leak across splits.
- Selects preprocessing mode, image size, and `K` strictly on the validation set before refitting on train + validation (`912` images) and evaluating once on the test set (`161` images).
- Generates evaluation tables, charts, and a print-ready A4 PDF report (`report/assignment_report.pdf`).

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

## Dataset download instructions

1. Configure Kaggle authentication locally (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`). Never commit credentials.
2. Download the dataset using the Kaggle CLI or from the [Kaggle dataset page](https://www.kaggle.com/dsv/9820160):

```bash
kaggle datasets download -d sanjaypate/devanagari-script-digital-44000-images-dataset
```

3. Extract the archive under `data/raw/` so `data/raw/DEVA-C-SET-CSV/DIGITS/CSV_FILES/0.csv` through `9.csv` exist.
4. Do not commit raw dataset files, archives, or API credentials.

See [DATASET.md](DATASET.md) for the complete dataset inspection summary.

## Folder structure

```text
.
├── data/
│   └── raw/                  # Local extracted Kaggle dataset (ignored by Git)
├── report/
│   ├── assignment_report.md  # Academic report in Markdown
│   └── assignment_report.pdf # Generated A4 PDF report (ignored by Git)
├── results/                  # Generated metrics, CSVs, and plots (ignored by Git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # CSV/image loading, deduplication, preprocessing
│   ├── train_evaluate.py     # Stratified splits, validation selection, test evaluation
│   └── generate_report.py    # Builds report/assignment_report.pdf from results/
├── tests/
│   └── test_pipeline.py      # Unit and integration tests
├── .gitignore
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── DATASET.md
├── EVALUATION.md
├── LICENSE
├── METHODOLOGY.md
├── PROJECT_STRUCTURE.md
├── README.md
└── requirements.txt
```

## Execution commands

Run the full experiment (compares `raw` and `centered` modes across `16 x 16` and `32 x 32` for `K = 1, 3, 5, 7, 9`, using `random_state = 42`):

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results
```

Generate the A4 PDF report from `results/`:

```bash
python -m src.generate_report
```

Run compilation check and tests:

```bash
python -m compileall src tests
python -m pytest -v
```

### Generated outputs in `results/`

- `results/metrics.json`: Full configuration, split counts, validation grid results, test metrics, per-class metrics, timings, and environment info.
- `results/dataset_inspection.json`: Raw vs. unique counts, excluded `CHAR` summary, dimensions, and corrupt-file checks.
- `results/k_comparison.csv`: Validation accuracy and timings across all 20 candidate configurations.
- `results/classification_report.txt`: Per-class precision, recall, F1-score, and support.
- `results/confusion_matrix.csv` and `results/confusion_matrix.png`: 10 x 10 confusion matrix.
- `results/sample_predictions.csv` and `results/prediction_samples.png`: Test predictions, nearest-neighbor distances, and error grid.
- `results/class_distribution.png`, `results/dataset_samples.png`, and `results/k_validation_curve.png`.

## Evaluation results

Measured on the 1,073 unique Devanagari digit images (split: 751 train, 161 validation, 161 held-out test; `random_state = 42`):

- **Selected configuration:** `centered`, `16 x 16` (256 features), `K = 3`, Euclidean distance, distance-weighted voting
- **Validation accuracy (selected):** `99.38%` (`raw` `16 x 16` peaked at `68.32%` for `K = 1`)
- **Test accuracy:** `95.03%` (`153 / 161` correct)
- **Macro precision:** `95.82%`
- **Macro recall:** `94.29%`
- **Macro F1-score:** `94.86%`
- **Weighted precision / recall / F1:** `95.29%` / `95.03%` / `95.01%`

## Confusion matrix preview

Actual test labels (rows) vs. predicted labels (columns) on the 161 test images:

| True \ Pred | 0 (०) | 1 (१) | 2 (२) | 3 (३) | 4 (४) | 5 (५) | 6 (६) | 7 (७) | 8 (८) | 9 (९) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0 (०)** | **7** | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| **1 (१)** | 0 | **16** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| **2 (२)** | 0 | 0 | **17** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3 (३)** | 0 | 0 | 0 | **15** | 0 | 0 | 0 | 0 | 0 | 0 |
| **4 (४)** | 0 | 0 | 0 | 0 | **15** | 1 | 0 | 0 | 0 | 0 |
| **5 (५)** | 0 | 1 | 0 | 0 | 0 | **18** | 0 | 0 | 0 | 0 |
| **6 (६)** | 0 | 0 | 0 | 0 | 0 | 0 | **19** | 0 | 0 | 0 |
| **7 (७)** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | **14** | 0 | 0 |
| **8 (८)** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **15** | 0 |
| **9 (९)** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **17** |

## Sample prediction examples

From `results/sample_predictions.csv`:

| Test Index | Source Row | True | Predicted | 1-NN Distance | Outcome |
|---:|---|:---:|:---:|---:|:---|
| 486 | `4.csv#row_724` | 4 (४) | 4 (४) | 2.6050 | Correct |
| 895 | `8.csv#row_424` | 8 (८) | 8 (८) | 1.5791 | Correct |
| 350 | `3.csv#row_450` | 3 (३) | 3 (३) | 3.4530 | Correct |
| 129 | `1.csv#row_530` | 1 (१) | 9 (९) | 2.9716 | Misclassified |
| 14 | `0.csv#row_112` | 0 (०) | 6 (६) | 2.9269 | Misclassified |
| 793 | `7.csv#row_372` | 7 (७) | 4 (४) | 4.5217 | Misclassified |

## Limitations

- Removing exact duplicate rows leaves 1,073 unique digit images (only 59 unique examples for digit `०`, 9 in the test split).
- Raw uncentered Euclidean distance achieves only 68.32% validation accuracy because digits vary in position and scale across the 65 x 65 canvas.
- Faint toolbar corner artifacts beyond 3 pixels in two test `०` images expanded their bounding boxes and caused `० -> ६` errors.
- KNN inference scales as O(N x D) with training set size, and digital canvas drawings may not generalize to scanned paper or camera photos.

## References

1. Sanjay E. Pate, *DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES*, Kaggle, DOI: `10.34740/KAGGLE/DSV/9820160`.
2. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21-27, 1967.
3. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825-2830, 2011.

## License and repository

Code is released under the [MIT License](LICENSE). The dataset listing specifies CC0 Public Domain.

GitHub repository: https://github.com/AkashKundu114/knn-devanagari-digit-recognition
