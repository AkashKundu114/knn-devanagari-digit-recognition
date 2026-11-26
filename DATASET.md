# Dataset

## Source

**DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES** by Sanjay E. Pate.

- Kaggle URL: https://www.kaggle.com/dsv/9820160
- Resolved Kaggle slug: `sanjaypate/devanagari-script-digital-44000-images-dataset`
- DOI: https://doi.org/10.34740/KAGGLE/DSV/9820160
- Dataset listing license: CC0: Public Domain
- Archive downloaded: `devanagari-script-digital-44000-images-dataset.zip` (70,590,540 bytes; extracted size 1,352,070,766 bytes under `data/raw/`)

## Actual inspected structure

Although the Kaggle description mentions 44,000 grayscale 65 x 65 JPEG images, the actual archive extracts as `DEVA-C-SET-CSV/` containing CSV matrices of flattened 65 x 65 grayscale pixels (4,225 integer columns in `[0, 255]`):

- `DEVA-C-SET-CSV/CHAR/`: Alphabet subset (`CSV_FILES/`: 35 CSV files, 32,855 rows including `111.csv` with 9 rows; `TRAIN_FILES/`: 34 files; `TEST_FILES/`: 34 files). All `CHAR/` files are excluded from this digit classification project.
- `DEVA-C-SET-CSV/DIGITS/`: Numeral subset (`0` through `9`):
  - `CSV_FILES/0.csv` to `9.csv`: 10 files with no header row, totaling **8,676 raw rows** (`0` missing/NaN values, `0` corrupt rows).
  - `TRAIN_FILES/` and `TEST_FILES/`: Pre-split files created by reading row 0 of `CSV_FILES/` as a header and saving an index column. These pre-split files are not used directly because duplicate rows cause near-100% overlap between `TRAIN_FILES` and `TEST_FILES`.

## Deduplication and per-class counts

SHA-256 hashing across `DIGITS/CSV_FILES/` identified **7,603 exact byte-identical duplicate rows** and **0 cross-class duplicates**, leaving **1,073 unique digit images**:

| Digit | Glyph | Raw CSV Rows | Duplicates Removed | Unique Images |
|---:|:---:|---:|---:|---:|
| 0 | ० | 525 | 466 | 59 |
| 1 | १ | 959 | 836 | 123 |
| 2 | २ | 899 | 784 | 115 |
| 3 | ३ | 899 | 801 | 98 |
| 4 | ४ | 899 | 789 | 110 |
| 5 | ५ | 899 | 773 | 126 |
| 6 | ६ | 899 | 773 | 126 |
| 7 | ७ | 899 | 802 | 97 |
| 8 | ८ | 899 | 799 | 100 |
| 9 | ९ | 899 | 780 | 119 |
| **Total** | **०-९** | **8,676** | **7,603** | **1,073** |

## Local setup

Download and extract the Kaggle archive under `data/raw/`. `src/data_loader.py` locates `DIGITS/CSV_FILES/` automatically (and also supports `DIGITS/<class-folder>/*.jpg|png` layouts). Raw data and Kaggle credentials are ignored by Git.

## Attribution

Sanjay E. Pate, *DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES*, Kaggle, DOI: `10.34740/KAGGLE/DSV/9820160`.
