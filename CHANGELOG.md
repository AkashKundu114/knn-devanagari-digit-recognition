# Changelog

## 2026-11-26 - Complete KNN Pipeline, Documentation, and CI

- Updated `src/data_loader.py` to load `DEVA-C-SET-CSV/DIGITS/CSV_FILES/` alongside image-folder layouts, handle corrupt rows/files safely, exclude `CHAR/`, and deduplicate exact SHA-256 duplicates (`8,676 -> 1,073` unique digit images).
- Updated `src/train_evaluate.py` to compare `raw` vs. `centered` preprocessing, `16 x 16` vs. `32 x 32` resolutions, `distance` vs. `uniform` weighting, and `K in {1, 3, 5, 7, 9}` on validation (`40` candidates), plus generate `classification_report.csv`, `dataset_inspection.json`, and evaluation plots.
- Executed full Kaggle evaluation (`centered 16 x 16, K = 3, distance` selected at `99.38%` validation accuracy; `95.03%` test accuracy, `94.86%` macro F1 on `161` test images).
- Added GitHub Actions workflows (`.github/workflows/ci.yml`, `manual-evaluation.yml`) and unit/integration tests in `tests/test_pipeline.py`.
- Synchronized `README.md`, `DATASET.md`, `METHODOLOGY.md`, and `EVALUATION.md`, added `report/`, `*.pptx`, and `src/generate_report.py` to `.gitignore`, and committed evaluation outputs and plots under `results/`.
- Extracted plotting helpers into `src/plots.py` and simplified `src/data_loader.py` and `src/train_evaluate.py` for readability.
- Removed `CONTRIBUTING.md`, `CITATION.cff`, and `PROJECT_STRUCTURE.md` to keep the repository minimal.
