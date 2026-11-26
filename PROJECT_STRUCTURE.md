# Project Structure

```text
.
├── data/
│   └── raw/                    # Local Kaggle dataset; ignored by Git
│       └── .gitkeep
├── report/
│   ├── assignment_report.md    # Academic assignment report (Markdown)
│   └── assignment_report.pdf   # Generated A4 PDF report; ignored by Git
├── results/                    # Generated evaluation outputs; ignored by Git
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV/image dataset loading, deduplication, preprocessing
│   ├── generate_report.py      # Builds report/assignment_report.pdf from results/
│   └── train_evaluate.py       # Stratified splitting, K selection, evaluation, plots
├── tests/
│   └── test_pipeline.py        # Unit and integration tests (pytest)
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
