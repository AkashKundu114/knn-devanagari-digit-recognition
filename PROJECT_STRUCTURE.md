# Project Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Python CI, pytest, link check, smoke test
│       └── manual-evaluation.yml      # Manual Kaggle evaluation workflow
├── data/
│   └── raw/                           # Local Kaggle dataset; ignored by Git
│       └── .gitkeep
├── results/                           # Generated evaluation outputs; ignored by Git
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # CSV/image loading, deduplication, preprocessing
│   └── train_evaluate.py              # Splits, grid search, evaluation, plots
├── tests/
│   └── test_pipeline.py               # Unit and integration tests (pytest)
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
