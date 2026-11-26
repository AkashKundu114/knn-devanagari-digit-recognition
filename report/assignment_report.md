# Assignment 3: KNN-Based Devanagari Handwritten Digit Recognition

**Student Name:** Akash Kundu  
**Course:** B.Tech Computer Science and Engineering  
**Roll Number:** 231001001449  
**Subject:** Machine Learning  
**Project Title:** Devanagari Handwritten Digit Recognition Using K-Nearest Neighbors

## 1. Introduction

Handwritten digit recognition is a practical image-classification problem. Different people write the same numeral in different styles, sizes and shapes. The objective is to classify an input handwritten Devanagari numeral into one of ten classes, from ० to ९, using K-Nearest Neighbors (KNN).

KNN is a supervised, instance-based learning algorithm. It predicts the class of a new sample by finding the K most similar training samples and using their labels to determine the prediction.

## 2. Dataset Description

The project uses the Kaggle dataset **DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES** by Sanjay E. Pate. The dataset listing describes 44,000 grayscale JPEG images representing 44 character classes: 34 alphabet classes and 10 numeral classes. The listed image resolution is 65 × 65 pixels. This project uses only the ten numeral folders and excludes the alphabet folders.

- Dataset: https://www.kaggle.com/dsv/9820160
- DOI: 10.34740/KAGGLE/DSV/9820160
- Licence shown on Kaggle: CC0: Public Domain

The actual number of digit images loaded is recorded in `results/metrics.json` after execution.

## 3. Algorithm Used: K-Nearest Neighbors

For this project:
1. Each image is converted to grayscale.
2. Each image is resized to 16 × 16 pixels.
3. The 256 pixel values are flattened into a one-dimensional feature vector.
4. Pixel values are normalized to the range 0 to 1.
5. Euclidean distance is used to find nearby training images.
6. Distance-weighted voting is used to predict the digit class.

The implementation tests K values 1, 3, 5, 7 and 9 on a validation set. The best validation K is then refitted using training plus validation samples and evaluated once on the held-out test set.

## 4. Implementation: All Steps

1. **Problem definition:** identify handwritten Devanagari numerals from ० to ९.
2. **Dataset collection:** download the public Kaggle dataset and extract it under `data/raw/`.
3. **Class selection:** use only the `DIGITS` directory; exclude `CHAR`.
4. **Image loading:** read images from the ten numeral class folders.
5. **Preprocessing:** convert to grayscale and resize to 16 × 16 pixels.
6. **Feature extraction:** flatten each image into 256 pixel features.
7. **Normalization:** scale pixel intensity values from 0–255 to 0–1.
8. **Data splitting:** create stratified training, validation and test subsets (70%, 15%, 15%).
9. **Model selection:** compare K = 1, 3, 5, 7 and 9 using validation accuracy.
10. **Model training:** refit KNN with the selected K on combined training and validation data.
11. **Prediction:** classify unseen test images.
12. **Evaluation:** calculate accuracy, macro precision, macro recall, macro F1-score and confusion matrix.
13. **Result storage:** save metrics, classification report, confusion matrix and sample predictions in `results/`.

## 5. Results and Evaluation

**Complete this section after running the project on the downloaded dataset. Do not submit estimated or fabricated scores.**

| Evaluation item | Measured result |
|---|---|
| Images loaded | Add value from `results/metrics.json` |
| Training images | Add value from `results/metrics.json` |
| Validation images | Add value from `results/metrics.json` |
| Test images | Add value from `results/metrics.json` |
| Selected K | Add value from `results/metrics.json` |
| Test accuracy | Add value from `results/metrics.json` |
| Macro precision | Add value from `results/metrics.json` |
| Macro recall | Add value from `results/metrics.json` |
| Macro F1-score | Add value from `results/metrics.json` |

Insert the generated `results/confusion_matrix.png` and selected sample predictions here. Per-class performance is available in `results/classification_report.txt`.

## 6. Conclusion

This project demonstrates how K-Nearest Neighbors can be applied to a real image-classification problem. It covers dataset collection, image preprocessing, feature representation, normalization, data splitting, model selection, prediction and evaluation. It uses a Devanagari numeral dataset rather than the commonly used MNIST dataset.

The final conclusion about model performance should be written after examining the measured test accuracy, macro F1-score and confusion matrix.

## Technologies Used

Python, NumPy, Pandas, Pillow, Matplotlib and Scikit-learn.

## GitHub Repository

https://github.com/AkashKundu114/knn-devanagari-digit-recognition
