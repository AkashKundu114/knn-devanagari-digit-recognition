# Devanagari Handwritten Digit Recognition Using KNN

A reproducible classical machine-learning project for classifying handwritten Devanagari numerals (०–९) with K-Nearest Neighbors (KNN).

## Project scope

- **Task:** ten-class image classification (० through ९).
- **Dataset:** [DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES](https://www.kaggle.com/dsv/9820160).
- **Subset:** the ten numeral classes only; alphabet classes are excluded.
- **Preprocessing:** grayscale, resize to 16 × 16, flatten to 256 features, scale pixels to [0, 1].
- **Classifier:** scikit-learn KNeighborsClassifier using Euclidean distance and distance-weighted voting.
- **Model selection:** validation comparison of K = 1, 3, 5, 7, 9.
- **Evaluation:** final held-out test set with accuracy, macro precision/recall/F1 and confusion matrix.

The Kaggle listing describes 44,000 grayscale JPEG images across 44 classes (34 alphabet classes and 10 numerals), at 65 × 65 pixels. The dataset page lists CC0: Public Domain. Raw dataset files are not included in this repository.

## Repository layout

```text
data/raw/                 Extracted dataset (not committed)
report/                   Assignment report (Markdown and PDF)
results/                  Generated evaluation outputs (not committed)
src/data_loader.py        Dataset loading and preprocessing
src/train_evaluate.py     Split, model selection, training and evaluation
requirements.txt          Python dependencies
```

## Setup

Python 3.10+ is recommended.

```bash
git clone https://github.com/AkashKundu114/knn-devanagari-digit-recognition.git
cd knn-devanagari-digit-recognition
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset preparation

1. Download the dataset from [Kaggle](https://www.kaggle.com/dsv/9820160).
2. Extract it under `data/raw/`.
3. Ensure the extracted archive contains a `DIGITS/` directory with ten class subfolders. The separate `CHAR/` alphabet directory is not used.
4. Do not commit the raw images, Kaggle credentials, or API tokens.

Dataset citation and details: see [DATASET.md](DATASET.md).

## Run

From the repository root:

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results
```

Optional parameters:

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results --image-size 16 --test-size 0.15 --validation-size 0.15 --random-state 42
```

The script creates stratified training, validation and test splits; chooses K using validation accuracy; refits on training + validation; then evaluates on the untouched test set.

## Outputs

- `results/metrics.json`
- `results/classification_report.txt`
- `results/confusion_matrix.png` and `.csv`
- `results/k_comparison.csv`
- `results/sample_predictions.csv`

No accuracy score is claimed in advance. Run the experiment on the downloaded dataset and report the measured results. See [EVALUATION.md](EVALUATION.md) and [report/assignment_report.md](report/assignment_report.md).

## Method and limitations

See [METHODOLOGY.md](METHODOLOGY.md). Flattened pixel features are a deliberately simple baseline and do not preserve all spatial structure. KNN inference can become expensive as the training set grows. Dataset performance may not generalize to scanned forms or camera-captured handwriting.

## License

The project code is released under the MIT License. The dataset has its own terms; the Kaggle listing identifies it as CC0. See [DATASET.md](DATASET.md) for attribution.

## Academic use

This repository is an independent KNN implementation using a Devanagari numeral dataset rather than the MNIST example in the supplied reference. Run the code and insert actual results into the assignment report before submission.
