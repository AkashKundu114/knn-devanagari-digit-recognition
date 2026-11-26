# Dataset

## Source

**DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES** by Sanjay E. Pate.

- Kaggle: https://www.kaggle.com/dsv/9820160
- DOI: https://doi.org/10.34740/KAGGLE/DSV/9820160
- Dataset listing licence: CC0: Public Domain
- Listed content: 44,000 grayscale JPEG images, 65 × 65 pixels, 44 classes.

## Subset used

This project uses only the ten Devanagari numeral classes (०–९) under the dataset's `DIGITS` directory. The 34 alphabet classes under `CHAR` are outside the assignment scope and are not loaded.

## Local setup

Download the archive from Kaggle and extract it under `data/raw/`. The loader searches recursively for a directory named `DIGITS`. It expects one class folder per numeral beneath that directory. If the downloaded archive uses different folder labels or layout, inspect it and adjust `LABEL_ALIASES` and the loader accordingly.

The raw dataset is intentionally excluded from Git. Do not upload Kaggle credentials or private tokens.

## Attribution

Sanjay E. Pate, *DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES*, Kaggle, DOI: 10.34740/KAGGLE/DSV/9820160.

The dataset's licence is separate from this repository's MIT licence. Refer to the Kaggle dataset page for its current terms and attribution requirements.
