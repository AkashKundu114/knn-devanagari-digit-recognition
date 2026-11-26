# Evaluation

Run the experiment from the repository root after extracting the Kaggle dataset into `data/raw/`:

```bash
python -m src.train_evaluate --data-dir data/raw --output-dir results
```

## Generated files

| File | Contents |
|---|---|
| `results/metrics.json` | Configuration, dataset counts, validation comparison, test metrics, per-class metrics, timings, environment |
| `results/dataset_inspection.json` | Raw vs. unique counts, excluded alphabet subset summary, corrupt file/row checks |
| `results/k_comparison.csv` | Validation accuracy and timings for all 40 `(preprocess_mode, image_size, weights, k)` candidates |
| `results/classification_report.txt` | Formatted per-class precision, recall, F1-score, and support |
| `results/classification_report.csv` | Machine-readable per-class precision, recall, F1-score, and support |
| `results/confusion_matrix.csv` | 10 x 10 confusion matrix counts |
| `results/confusion_matrix.png` | Visual confusion matrix heatmap |
| `results/k_validation_curve.png` | Validation accuracy vs. K across preprocessing, size, and weighting |
| `results/class_distribution.png` | Raw rows vs. deduplicated unique image counts per digit class |
| `results/dataset_samples.png` | 10 x 5 grid of representative unique digit images |
| `results/prediction_samples.png` | Grid of misclassified and correct test predictions |
| `results/sample_predictions.csv` | All 161 test predictions with nearest-neighbor distances |

## Measured results (Kaggle dataset, `random_state = 42`)

- **Images used:** `1,073` unique images (from `8,676` raw digit rows; `7,603` exact duplicates removed)
- **Split sizes:** `751` train / `161` validation / `912` refit train+val / `161` held-out test
- **Selected configuration:** `centered`, `16 x 16` (`256` features), `K = 3`, Euclidean distance, `distance` weighting
- **Selected validation accuracy:** `99.38%` (tied between `distance` and `uniform` at `K = 3`; `distance` selected by tie-break)
- **Test accuracy:** `95.03%` (`153 / 161`)
- **Macro precision / recall / F1-score:** `95.82%` / `94.29%` / `94.86%`
- **Timings:** data loading `11.15 s`, validation search (40 candidates) `3.21 s`, final fit `0.0011 s`, test inference `0.0280 s`

## Confusion matrix and per-class notes

- Classes `2 (२)`, `3 (३)`, and `8 (८)` achieved `1.0000` F1-score on the held-out test set.
- Class `6 (६)` achieved `1.0000` recall (`0.8636` precision).
- Out of `8` test errors: two `0 (०)` images were predicted as `6 (६)` due to faint toolbar artifacts beyond the 3-pixel border shifting their bounding boxes, two errors swapped structurally similar loops `1 (१)` and `9 (९)`, and four isolated errors occurred (`1->6`, `4->5`, `5->1`, `7->4`).
