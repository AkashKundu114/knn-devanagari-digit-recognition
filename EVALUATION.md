# Evaluation

Run the project after downloading and extracting the Kaggle dataset:

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results
```

The script writes:

| File | Contents |
|---|---|
| `metrics.json` | Image counts, split sizes, selected K, test accuracy, macro precision, macro recall and macro F1 |
| `k_comparison.csv` | Validation accuracy for each candidate K |
| `classification_report.txt` | Per-class precision, recall and F1 |
| `confusion_matrix.csv` | Confusion matrix values |
| `confusion_matrix.png` | Visual confusion matrix |
| `sample_predictions.csv` | Example actual and predicted labels |

## Reporting rule

Do not enter estimated or invented scores in the assignment report. Copy the measured values from `metrics.json` only after a successful run. Include the generated confusion matrix and discuss classes with the largest off-diagonal counts.

The dataset has not been executed as part of repository scaffolding; reported performance remains to be measured locally.
