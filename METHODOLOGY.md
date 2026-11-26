# Methodology

## Pipeline

1. Locate `DIGITS/CSV_FILES/` under `data/raw/` and exclude `CHAR/`.
2. Read `0.csv` through `9.csv` (`header=None`), validate 4,225-column 65 x 65 integer pixel rows in `[0, 255]`, and skip any malformed rows safely.
3. Deduplicate exact image matrices via SHA-256 hashing (`8,676` raw rows -> `1,073` unique images) to prevent train-test data leakage.
4. Create disjoint stratified splits with `random_state = 42`: 70% training (`751`), 15% validation (`161`), and 15% held-out test (`161`).
5. Generate normalized `[0, 1]` feature vectors across two preprocessing modes (`raw` and `centered`) and two image resolutions (`16 x 16` = 256 features, `32 x 32` = 1,024 features):
   - `raw`: direct Lanczos resize from 65 x 65 to `S x S`.
   - `centered`: 3-pixel border cleanup, tight bounding-box crop on ink pixels (`< 200`), symmetric square padding, and Lanczos resize to `(S - 2) x (S - 2)` centered on an `S x S` white canvas.
6. Fit `KNeighborsClassifier(metric="euclidean")` across `weights in {"distance", "uniform"}` and `K in {1, 3, 5, 7, 9}` for every `(mode, image_size)` pair on the validation split (`40` total candidates).
7. Select the best `(mode, image_size, weights, K)` strictly by validation accuracy (`centered`, `16 x 16`, `distance`, `K = 3`, `99.38%`).
8. Refit the selected model on combined training + validation (`912` samples) and evaluate once on the held-out test split (`161` samples).

## Classifier

KNN classifies a query vector `q` by computing Euclidean distance `d(q, x_i) = sqrt(sum_j (q_j - x_{i,j})^2)` to all stored training vectors. Under `uniform` voting, each of the `K` nearest neighbors casts an equal vote; under `distance` voting, each neighbor receives weight `1 / d(q, x_i)`.

## Reproducibility

Splitting is deterministic (`random_state = 42`), per-image preprocessing fits no global statistics across splits, and index-intersection assertions confirm zero overlap between train, validation, and test sets.

## Limitations

- Raw uncentered pixels are sensitive to translation and scale differences on the drawing canvas (`69.57%` max raw validation accuracy vs. `99.38%` centered).
- Prediction time scales linearly with training set size and feature count.
- Canvas drawings do not establish performance on paper scans or camera photographs.
