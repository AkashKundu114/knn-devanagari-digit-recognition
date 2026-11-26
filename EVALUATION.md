# Evaluation

Run the experiment from the repository root after extracting the Kaggle dataset into `data/raw/`:

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results
```

To regenerate the A4 PDF assignment report:

```bash
python -m src.generate_report
```

## Generated files

| File | Contents |
|---|---|
| `results/metrics.json` | Configuration, dataset counts, validation comparison, test metrics, per-class metrics, timings, environment |
| `results/dataset_inspection.json` | Raw vs. unique counts, excluded alphabet subset summary, corrupt file/row checks |
| `results/k_comparison.csv` | Validation accuracy and timings for all 20 `(preprocess_mode, image_size, k)` candidates |
| `results/classification_report.txt` | Per-class precision, recall, F1-score, and support |
| `results/confusion_matrix.csv` | 10 x 10 confusion matrix counts |
| `results/confusion_matrix.png` | Visual confusion matrix heatmap |
| `results/k_validation_curve.png` | Validation accuracy vs. K for `centered` and `raw` at `16 x 16` and `32 x 32` |
| `results/class_distribution.png` | Raw rows vs. deduplicated unique image counts per digit class |
| `results/dataset_samples.png` | 10 x 5 grid of representative unique digit images |
| `results/prediction_samples.png` | Grid of misclassified and correct test predictions |
| `results/sample_predictions.csv` | All 161 test predictions with nearest-neighbor distances |

## Measured results (Kaggle dataset, `random_state = 42`)

- **Images used:** 1,073 unique images (from 8,676 raw digit rows; 7,603 exact duplicates removed)
- **Split sizes:** 751 train / 161 validation / 912 refit train+val / 161 held-out test
- **Selected configuration:** `centered`, `16 x 16` (256 features), `K = 3`
- **Selected validation accuracy:** `99.38%`
- **Test accuracy:** `95.03%` (`153 / 161`)
- **Macro precision / recall / F1-score:** `95.82%` / `94.29%` / `94.86%`
