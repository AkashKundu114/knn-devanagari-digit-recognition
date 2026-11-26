# Methodology

## Pipeline

1. Locate the `DIGITS` folder and ignore the alphabet subset.
2. Read images from each numeral class folder.
3. Convert each image to grayscale.
4. Resize each image to 16 × 16 pixels.
5. Convert the 256 pixel intensities into a flattened feature vector.
6. Normalize intensity values from 0–255 to 0–1.
7. Create stratified training, validation and test partitions (default 70/15/15).
8. Fit candidate KNN classifiers for K = 1, 3, 5 and 7 and 9.
9. Select K by validation accuracy.
10. Refit the selected model on the combined training and validation samples.
11. Evaluate once on the held-out test partition.
12. Save aggregate metrics, per-class metrics, confusion matrix and sample predictions.

## Classifier

The implementation uses Euclidean distance and distance-weighted neighbour voting. For a query vector, the model finds its K closest training vectors and predicts based on the labels of those neighbours, giving closer neighbours greater voting weight.

## Reproducibility

The split uses a configurable random seed (default 42). Stratification maintains class proportions across partitions. The test set is not used for hyperparameter selection.

## Limitations

Flattening pixels discards explicit spatial relationships. KNN stores training samples and can have high prediction cost on large datasets. Results depend on the downloaded dataset's actual folder structure and quality. Generalization to paper scans or camera images is not established by this experiment.
