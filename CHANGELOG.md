# Changelog

## v1.0.0 - Complete KNN Devanagari Digit Recognition Project

- Updated `src/data_loader.py` to support the real Kaggle `DEVA-C-SET-CSV/DIGITS/CSV_FILES/` format alongside image-folder layouts, with corrupt-row/file checks and SHA-256 deduplication (`8,676 -> 1,073` unique digit images).
- Added `raw` and `centered` preprocessing modes and compared `16 x 16` vs. `32 x 32` resolutions across `K = 1, 3, 5, 7, 9` on validation only.
- Executed the full Kaggle experiment (`centered 16 x 16, K = 3` selected at `99.38%` validation accuracy; `95.03%` test accuracy, `94.86%` macro F1 on 161 held-out test images).
- Added `src/generate_report.py` and updated `report/assignment_report.md` and `report/assignment_report.pdf`.
- Added unit and integration tests in `tests/test_pipeline.py` (18 passing tests).
- Updated project documentation (`README.md`, `DATASET.md`, `METHODOLOGY.md`, `EVALUATION.md`, `PROJECT_STRUCTURE.md`, `CONTRIBUTING.md`).
