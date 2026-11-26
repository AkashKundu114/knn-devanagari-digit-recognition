# Project Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Python CI, pytest, synthetic smoke test
│       ├── manual-evaluation.yml      # Manual Kaggle evaluation workflow
│       └── report-check.yml           # LaTeX source and Markdown link check
├── data/
│   └── raw/                           # Local Kaggle dataset; ignored by Git
│       └── .gitkeep
├── report/
│   ├── assignment_report.md           # Concise Markdown report matching the PDF
│   ├── assignment_report.pdf          # Compiled 3-page A4 LaTeX PDF; ignored by Git
│   └── latex/
│       ├── main.tex                   # Main LaTeX assignment report source
│       ├── generated_results.tex      # Auto-generated macros/tables from results/
│       └── figures/                   # Report figures copied from results/
├── results/                           # Generated evaluation outputs; ignored by Git
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # CSV/image loading, deduplication, preprocessing
│   ├── generate_report.py             # Generates LaTeX tables and compiles PDF
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
