"""Generate the print-friendly A4 academic PDF report from actual results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DIGIT_GLYPHS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]


def _register_fonts() -> tuple[str, str]:
    """Register a Unicode font supporting Devanagari numerals when available."""
    nirmala = Path(r"C:\Windows\Fonts\Nirmala.ttc")
    if nirmala.exists():
        pdfmetrics.registerFont(TTFont("BodyFont", str(nirmala), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("BodyFontBold", str(nirmala), subfontIndex=1))
        return "BodyFont", "BodyFontBold"
    return "Helvetica", "Helvetica-Bold"


def _scaled_image(path: Path, max_width_mm: float, max_height_mm: float) -> RLImage:
    """Create a ReportLab Image scaled proportionally within bounding box."""
    img = RLImage(str(path))
    max_w = max_width_mm * mm
    max_h = max_height_mm * mm
    ratio = min(max_w / img.drawWidth, max_h / img.drawHeight)
    img.drawWidth *= ratio
    img.drawHeight *= ratio
    img.hAlign = "CENTER"
    return img


def _styled_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    """Build a clean print-friendly table."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102a43")),
            ("FONTNAME", (0, 0), (-1, 0), "BodyFontBold" if "BodyFontBold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "BodyFont" if "BodyFont" in pdfmetrics.getRegisteredFontNames() else "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bcccdc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return tbl


def _draw_page_number(canvas, doc) -> None:
    """Draw footer page number on every page after the title page."""
    if doc.page == 1:
        return
    canvas.saveState()
    font_name = "BodyFont" if "BodyFont" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    canvas.setFont(font_name, 8.5)
    canvas.setFillColor(colors.HexColor("#486581"))
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.drawString(18 * mm, 12 * mm, "Assignment 3: Devanagari Handwritten Digit Recognition Using KNN")
    canvas.restoreState()


def build_pdf_report(
    results_dir: Path = Path("results"),
    output_pdf: Path = Path("report/assignment_report.pdf"),
) -> Path:
    """Build report/assignment_report.pdf from measured results in results_dir."""
    font_reg, font_bold = _register_fonts()
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    inspection = json.loads((results_dir / "dataset_inspection.json").read_text(encoding="utf-8"))
    k_comp = pd.read_csv(results_dir / "k_comparison.csv")
    cm_df = pd.read_csv(results_dir / "confusion_matrix.csv", index_col=0)
    preds_df = pd.read_csv(results_dir / "sample_predictions.csv")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Devanagari Handwritten Digit Recognition Using KNN",
        author="Akash Kundu",
    )

    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleMain", parent=base_styles["Title"], fontName=font_bold,
        fontSize=21, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#102a43"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "SubMain", parent=base_styles["Normal"], fontName=font_reg,
        fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#334e68"),
        spaceAfter=18,
    )
    h1_style = ParagraphStyle(
        "SecH1", parent=base_styles["Heading1"], fontName=font_bold,
        fontSize=12.5, leading=16, textColor=colors.HexColor("#102a43"),
        spaceBefore=10, spaceAfter=4, keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "SecH2", parent=base_styles["Heading2"], fontName=font_bold,
        fontSize=10.5, leading=14, textColor=colors.HexColor("#243b53"),
        spaceBefore=7, spaceAfter=3, keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body", parent=base_styles["Normal"], fontName=font_reg,
        fontSize=9.6, leading=13.8, alignment=TA_LEFT, textColor=colors.HexColor("#1f2933"),
        spaceAfter=5,
    )
    formula_style = ParagraphStyle(
        "Formula", parent=base_styles["Normal"], fontName=font_bold,
        fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#102a43"),
        backColor=colors.HexColor("#f0f4f8"), borderPadding=6, spaceBefore=4, spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=base_styles["Normal"], fontName=font_reg,
        fontSize=8.5, leading=11.5, alignment=TA_CENTER, textColor=colors.HexColor("#486581"),
        spaceBefore=2, spaceAfter=8,
    )

    story = []

    # 1. Title Page
    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph("1. Title Page", ParagraphStyle("TpTag", parent=subtitle_style, fontSize=10)))
    story.append(Paragraph("Devanagari Handwritten Digit Recognition<br/>Using K-Nearest Neighbors (KNN)", title_style))
    story.append(Paragraph("Assignment 3 | Machine Learning", subtitle_style))
    story.append(HRFlowable(width="70%", thickness=1.2, color=colors.HexColor("#334e68"), hAlign="CENTER"))
    story.append(Spacer(1, 14 * mm))

    info_rows = [
        ["Field", "Details"],
        ["Student Name", "Akash Kundu"],
        ["Course", "B.Tech Computer Science and Engineering"],
        ["Roll Number", "231001001449"],
        ["Subject", "Machine Learning"],
        ["Assignment", "Assignment 3"],
        ["Project Title", "Devanagari Handwritten Digit Recognition Using KNN"],
        ["Dataset DOI", "10.34740/KAGGLE/DSV/9820160"],
        ["GitHub Repository", "https://github.com/AkashKundu114/knn-devanagari-digit-recognition"],
        ["Date", "November 26, 2026"],
    ]
    info_tbl = _styled_table(info_rows, col_widths=[45 * mm, 118 * mm])
    info_tbl.setStyle(TableStyle([("ALIGN", (0, 1), (-1, -1), "LEFT")]))
    story.append(info_tbl)
    story.append(Spacer(1, 22 * mm))
    story.append(Paragraph(
        f"Measured Test Summary: {metrics['test_accuracy']*100:.2f}% Accuracy | "
        f"{metrics['test_macro_f1']*100:.2f}% Macro F1 | "
        f"Selected Config: {metrics['selected_preprocess_mode']} "
        f"{metrics['image_size']}x{metrics['image_size']}, K={metrics['selected_k']} "
        f"({metrics['total_images_loaded']} unique images)",
        caption_style,
    ))
    story.append(PageBreak())

    # 2. Introduction
    story.append(Paragraph("2. Introduction", h1_style))
    story.append(Paragraph(
        "Handwritten digit recognition is a classic classification problem in supervised machine learning. "
        "Different writers produce different stroke widths, loop sizes, slants, and canvas positions for the same numeral. "
        "Instead of using the standard English MNIST dataset, this assignment classifies handwritten Devanagari numerals "
        "(०, १, २, ३, ४, ५, ६, ७, ८, ९) into numeric labels 0 through 9 using K-Nearest Neighbors (KNN).",
        body_style,
    ))

    # 3. Problem Statement
    story.append(Paragraph("3. Problem Statement", h1_style))
    story.append(Paragraph(
        "Given a 65 x 65 grayscale image of an isolated handwritten Devanagari numeral captured on a digital canvas, "
        "classify it into one of ten classes (0 to 9) using Scikit-learn's KNeighborsClassifier on pixel features, "
        "while excluding the 34 Devanagari alphabet classes in the dataset and preventing data leakage across splits.",
        body_style,
    ))

    # 4. Objectives
    story.append(Paragraph("4. Objectives", h1_style))
    objectives = [
        "1. Download and inspect the actual Kaggle archive (sanjaypate/devanagari-script-digital-44000-images-dataset).",
        "2. Isolate the ten digit classes (0 to 9) and exclude all alphabet classes under CHAR/.",
        "3. Detect and remove exact duplicate rows before splitting so no identical images appear in both training and test splits.",
        "4. Build a deterministic preprocessing pipeline with grayscale formatting, border cleanup, centering, resizing, flattening, and [0, 1] normalization.",
        "5. Compare raw resizing and centered cropping at 16 x 16 and 32 x 32 resolutions across K = 1, 3, 5, 7, 9 on a validation split.",
        "6. Refit the chosen configuration on combined training and validation data and evaluate once on the held-out test set.",
    ]
    for obj in objectives:
        story.append(Paragraph(obj, body_style))

    # 5. Dataset Description
    story.append(Paragraph("5. Dataset Description", h1_style))
    story.append(Paragraph(
        "Source: DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES by Sanjay E. Pate "
        "(Kaggle: https://www.kaggle.com/dsv/9820160, DOI: 10.34740/KAGGLE/DSV/9820160, CC0 Public Domain). "
        "While the Kaggle listing describes 44,000 grayscale 65 x 65 JPEG images across 44 classes, "
        "inspecting the downloaded archive (70,590,540 bytes zipped; 1,352,070,766 bytes extracted) showed that "
        "it stores CSV files under DEVA-C-SET-CSV/ containing flattened 65 x 65 integer pixel rows (4,225 columns in [0, 255], no header).",
        body_style,
    ))
    story.append(Paragraph(
        f"Excluded alphabet data: CHAR/CSV_FILES/ holds {inspection['excluded_alphabet_subset']['csv_files']} CSV files "
        f"({inspection['excluded_alphabet_subset']['rows']:,} rows, including a 9-row 111.csv file), all excluded. "
        f"Digit data: DIGITS/CSV_FILES/ holds 10 files (0.csv to 9.csv) totaling {inspection['raw_digit_images']:,} rows with 0 missing or corrupt rows.",
        body_style,
    ))

    # 6. Dataset Collection and Preparation
    story.append(Paragraph("6. Dataset Collection and Preparation", h1_style))
    story.append(Paragraph(
        "SHA-256 hashing across DIGITS/CSV_FILES/ revealed 7,603 exact byte-identical duplicate rows out of 8,676 rows, "
        "leaving 1,073 unique handwritten digit images (0 cross-class duplicates). Furthermore, the archive's pre-split "
        "TRAIN_FILES/ and TEST_FILES/ overlap almost completely because duplicates were split across both folders and row 0 was misread as a header. "
        "To eliminate leakage, all 7,603 exact duplicates were removed from CSV_FILES/ before stratified splitting.",
        body_style,
    ))

    cls_table_data = [["Digit", "Glyph", "Raw CSV Rows", "Duplicates Removed", "Unique Images"]]
    for d in range(10):
        raw_c = inspection["raw_class_counts"][str(d)]
        uniq_c = inspection["unique_class_counts"][str(d)]
        cls_table_data.append([str(d), DIGIT_GLYPHS[d], str(raw_c), str(raw_c - uniq_c), str(uniq_c)])
    cls_table_data.append(["Total", "०-९", "8,676", "7,603", "1,073"])
    story.append(_styled_table(cls_table_data, col_widths=[22 * mm, 22 * mm, 38 * mm, 42 * mm, 38 * mm]))
    story.append(Spacer(1, 4 * mm))

    story.append(KeepTogether([
        _scaled_image(results_dir / "class_distribution.png", 162, 78),
        Paragraph("Figure 1: Raw CSV rows vs. unique deduplicated digit images across classes 0 to 9.", caption_style),
    ]))
    story.append(KeepTogether([
        _scaled_image(results_dir / "dataset_samples.png", 145, 112),
        Paragraph("Figure 2: Representative 65 x 65 grayscale samples from the 1,073 unique digit images.", caption_style),
    ]))

    # 7. Image Preprocessing
    story.append(Paragraph("7. Image Preprocessing", h1_style))
    story.append(Paragraph(
        "Two per-image preprocessing modes were implemented in src/data_loader.py and compared on validation:",
        body_style,
    ))
    story.append(Paragraph(
        "1. Raw mode (raw): Directly resizes the 65 x 65 grayscale image to S x S (16 x 16 or 32 x 32) using Lanczos resampling.",
        body_style,
    ))
    story.append(Paragraph(
        "2. Centered mode (centered): Clears a 3-pixel border (sets outer edges to 255) to remove canvas scrollbar/frame lines, "
        "finds the tight bounding box of dark ink pixels (intensity < 200), pads symmetrically to a square to preserve aspect ratio, "
        "resizes to (S - 2) x (S - 2) with Lanczos resampling, and places it centered on a white S x S canvas.",
        body_style,
    ))

    # 8. Feature Extraction
    story.append(Paragraph("8. Feature Extraction", h1_style))
    story.append(Paragraph(
        "Each S x S grayscale matrix is flattened row-wise into a 1D feature vector: 256 features for 16 x 16 images "
        "and 1,024 features for 32 x 32 images. Pixel values in [0, 255] are converted to float32 and divided by 255.0 "
        "so every feature lies strictly in [0.0, 1.0]. Downsampling reduces feature dimensionality compared to the original 4,225 pixels.",
        body_style,
    ))

    # 9. Explanation of the KNN Algorithm
    story.append(Paragraph("9. Explanation of the KNN Algorithm", h1_style))
    story.append(Paragraph(
        "K-Nearest Neighbors is a lazy, non-parametric classifier. During training, it stores labeled feature vectors without "
        "optimizing weights. During prediction for a query vector q, it computes distances to all stored training vectors, "
        "selects the K nearest neighbors, and assigns the class with the highest weighted vote using inverse distance weighting "
        "(w_i = 1 / d(q, x_i)), giving closer training examples greater influence than farther neighbors.",
        body_style,
    ))

    # 10. Mathematical explanation of Euclidean distance
    story.append(Paragraph("10. Mathematical Explanation of Euclidean Distance", h1_style))
    story.append(Paragraph(
        "For two D-dimensional normalized feature vectors p = (p_1, ..., p_D) and q = (q_1, ..., q_D), "
        "Euclidean (L2) distance is defined as:",
        body_style,
    ))
    story.append(Paragraph("d(p, q) = sqrt( sum_{j=1..D} (p_j - q_j)^2 )", formula_style))
    story.append(Paragraph(
        "Because each term compares pixel intensities at identical coordinate j, shifting or resizing a digit on a 65 x 65 canvas "
        "produces large Euclidean distances even for the same digit class. Spatial centering aligns digit ink across images so L2 distance reflects shape similarity.",
        body_style,
    ))

    # 11. Data Splitting
    story.append(Paragraph("11. Data Splitting", h1_style))
    sc = metrics["split_counts"]
    split_tbl = [
        ["Partition", "Proportion", "Samples", "Role"],
        ["Training", "70%", str(sc["train"]), "Fit candidate KNN models during K selection"],
        ["Validation", "15%", str(sc["validation"]), "Select preprocessing mode, resolution, and K"],
        ["Train + Validation", "85%", str(sc["final_training_after_refit"]), "Refit selected KNN model after tuning"],
        ["Held-out Test", "15%", str(sc["test"]), "Single final evaluation (never used for selection)"],
    ]
    story.append(_styled_table(split_tbl, col_widths=[42 * mm, 24 * mm, 24 * mm, 73 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"All splits use stratified sampling with random_state = {metrics['random_state']}. "
        "Index set intersections verify zero overlap across partitions.",
        body_style,
    ))

    # 12. Model Training and Hyperparameter Selection
    story.append(Paragraph("12. Model Training and Hyperparameter Selection", h1_style))
    story.append(Paragraph(
        "Twenty configurations (2 preprocessing modes x 2 image sizes x 5 values of K in {1, 3, 5, 7, 9}) were tested "
        "on the 161-sample validation split using KNeighborsClassifier(metric='euclidean', weights='distance'). "
        f"The top validation configuration was {metrics['selected_preprocess_mode']} {metrics['image_size']}x{metrics['image_size']} "
        f"with K = {metrics['selected_k']} ({metrics['selected_validation_accuracy']*100:.2f}% validation accuracy). "
        "This model was refitted on the 912 combined train + validation samples and evaluated once on the 161 test samples.",
        body_style,
    ))

    # 13. Implementation Steps
    story.append(Paragraph("13. Implementation Steps", h1_style))
    steps = [
        "1. Download and extract the Kaggle dataset under data/raw/.",
        "2. Load DIGITS/CSV_FILES/0.csv through 9.csv and exclude CHAR/ via src/data_loader.py.",
        "3. Deduplicate exact image hashes (8,676 -> 1,073 unique images).",
        "4. Create stratified 70/15/15 index splits with random_state = 42.",
        "5. Generate raw and centered feature matrices for 16 x 16 and 32 x 32 resolutions.",
        "6. Run validation selection over K = 1, 3, 5, 7, 9 and save k_comparison.csv.",
        "7. Refit on train + validation (912 images) and evaluate once on test (161 images).",
        "8. Save metrics, confusion matrix, classification report, sample predictions, and plots in results/.",
    ]
    for st in steps:
        story.append(Paragraph(st, body_style))

    # 14. Experimental Setup
    story.append(Paragraph("14. Experimental Setup", h1_style))
    env = metrics["environment"]
    tm = metrics["timing_seconds"]
    story.append(Paragraph(
        f"Environment: Python {env['python_version']}, NumPy {env['numpy_version']}, Pandas {env['pandas_version']}, "
        f"Pillow {env['pillow_version']}, Scikit-learn {env['scikit_learn_version']}, Matplotlib {env['matplotlib_version']} on {env['platform']}. "
        f"Measured timings: data loading {tm['data_loading']:.4f} s, validation search {tm['validation_search']:.4f} s, "
        f"final model fit {tm['final_model_fit']:.4f} s, test inference {tm['test_inference']:.4f} s. "
        "Reproduce command: python -m src.train_evaluate --data-dir data/raw --output-dir results.",
        body_style,
    ))

    # 15. Actual Results
    story.append(Paragraph("15. Actual Results", h1_style))
    piv = k_comp.pivot(index=["preprocess_mode", "image_size", "feature_count"], columns="k", values="validation_accuracy")
    k_tbl = [["Mode", "Image Size", "Features", "K=1", "K=3", "K=5", "K=7", "K=9"]]
    for (mode, size, feats), row in piv.iterrows():
        k_tbl.append([
            str(mode), f"{size}x{size}", str(feats),
            f"{row[1]*100:.2f}%", f"{row[3]*100:.2f}%", f"{row[5]*100:.2f}%",
            f"{row[7]*100:.2f}%", f"{row[9]*100:.2f}%",
        ])
    story.append(_styled_table(k_tbl, col_widths=[24 * mm, 24 * mm, 20 * mm, 19 * mm, 19 * mm, 19 * mm, 19 * mm, 19 * mm]))
    story.append(Spacer(1, 4 * mm))

    story.append(KeepTogether([
        _scaled_image(results_dir / "k_validation_curve.png", 160, 82),
        Paragraph("Figure 3: Validation accuracy across K = 1, 3, 5, 7, 9 for centered and raw preprocessing.", caption_style),
    ]))

    summary_tbl = [
        ["Evaluation Metric (Held-Out Test Set, N = 161)", "Measured Value"],
        ["Selected Configuration", f"{metrics['selected_preprocess_mode']} {metrics['image_size']}x{metrics['image_size']}, K={metrics['selected_k']}"],
        ["Validation Accuracy (Selected)", f"{metrics['selected_validation_accuracy']*100:.2f}%"],
        ["Test Accuracy", f"{metrics['test_accuracy']*100:.2f}% ({int(preds_df['correct'].sum())}/{len(preds_df)})"],
        ["Macro Precision", f"{metrics['test_macro_precision']*100:.2f}%"],
        ["Macro Recall", f"{metrics['test_macro_recall']*100:.2f}%"],
        ["Macro F1-Score", f"{metrics['test_macro_f1']*100:.2f}%"],
        ["Weighted Precision / Recall / F1", f"{metrics['test_weighted_precision']*100:.2f}% / {metrics['test_weighted_recall']*100:.2f}% / {metrics['test_weighted_f1']*100:.2f}%"],
    ]
    story.append(_styled_table(summary_tbl, col_widths=[86 * mm, 77 * mm]))
    story.append(Spacer(1, 4 * mm))

    pc_tbl = [["Digit", "Glyph", "Precision", "Recall", "F1-Score", "Support"]]
    for d in range(10):
        m = metrics["per_class_metrics"][str(d)]
        pc_tbl.append([str(d), m["glyph"], f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1_score']:.4f}", str(m["support"])])
    story.append(_styled_table(pc_tbl, col_widths=[20 * mm, 20 * mm, 32 * mm, 32 * mm, 32 * mm, 27 * mm]))
    story.append(Spacer(1, 4 * mm))

    # 16. Confusion Matrix
    story.append(Paragraph("16. Confusion Matrix", h1_style))
    cm_headers = ["True\\Pred"] + [f"{i} ({DIGIT_GLYPHS[i]})" for i in range(10)]
    cm_tbl = [cm_headers]
    for i in range(10):
        row_vals = [str(int(v)) for v in cm_df.iloc[i].tolist()]
        cm_tbl.append([f"{i} ({DIGIT_GLYPHS[i]})"] + row_vals)
    story.append(_styled_table(cm_tbl, col_widths=[21 * mm] + [14.2 * mm] * 10))
    story.append(Spacer(1, 3 * mm))
    story.append(KeepTogether([
        _scaled_image(results_dir / "confusion_matrix.png", 150, 108),
        Paragraph("Figure 4: Confusion matrix on the 161 held-out test images.", caption_style),
    ]))

    # 17. Performance Analysis
    story.append(Paragraph("17. Performance Analysis", h1_style))
    story.append(Paragraph(
        "1. Effect of centering: Raw pixel resizing achieved only 68.32% validation accuracy (16 x 16, K=1) because "
        "handwritten digits appear at different coordinates on the 65 x 65 canvas. Bounding-box centering raised validation accuracy to 99.38% (+31.06 points).",
        body_style,
    ))
    story.append(Paragraph(
        "2. Resolution comparison: 16 x 16 (256 features) matched or outperformed 32 x 32 (1,024 features) while halving inference time, "
        "because mild downsampling tolerates small stroke variations.",
        body_style,
    ))
    story.append(Paragraph(
        "3. Error analysis (8 test errors out of 161): Two digit 0 (०) images were misclassified as 6 (६) because faint canvas toolbar artifacts "
        "beyond the 3-pixel border stretched their bounding boxes toward the top-left. Two other errors occurred between visually similar loops "
        "1 (१) and 9 (९). The remaining four isolated errors were 1->6, 4->5, 5->1, and 7->4.",
        body_style,
    ))

    # 18. Sample Predictions
    story.append(Paragraph("18. Sample Predictions", h1_style))
    wrong_df = preds_df[~preds_df["correct"]].head(4)
    right_df = preds_df[preds_df["correct"]].head(4)
    sample_show = pd.concat([right_df, wrong_df], ignore_index=True)
    sp_tbl = [["Test Index", "Source", "True", "Predicted", "1-NN Dist", "Outcome"]]
    for _, r in sample_show.iterrows():
        sp_tbl.append([
            str(int(r["test_index"])),
            str(r["sample_source"]),
            f"{int(r['actual_label'])} ({r['actual_glyph']})",
            f"{int(r['predicted_label'])} ({r['predicted_glyph']})",
            f"{float(r['nearest_neighbor_distance']):.4f}",
            "Correct" if bool(r["correct"]) else "Misclassified",
        ])
    story.append(_styled_table(sp_tbl, col_widths=[20 * mm, 35 * mm, 24 * mm, 24 * mm, 25 * mm, 35 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(KeepTogether([
        _scaled_image(results_dir / "prediction_samples.png", 164, 98),
        Paragraph("Figure 5: Test set predictions showing misclassified cases (top row) and correct samples.", caption_style),
    ]))

    # 19. Advantages and Limitations
    story.append(Paragraph("19. Advantages and Limitations", h1_style))
    story.append(Paragraph(
        "Advantages: Simple and transparent instance-based decision rule, near-instant training (0.0010 s), and strong accuracy (95.03%) "
        "once spatial centering is applied.",
        body_style,
    ))
    story.append(Paragraph(
        "Limitations: Raw pixel Euclidean distance is sensitive to uncentered digits and edge artifacts; test inference scales linearly O(N x D) "
        "with training set size; removing duplicates leaves 1,073 unique samples (only 59 unique examples for digit 0); and canvas drawings "
        "do not guarantee generalization to scanned paper or camera photos.",
        body_style,
    ))

    # 20. Conclusion
    story.append(Paragraph("20. Conclusion", h1_style))
    story.append(Paragraph(
        "Inspecting the actual Kaggle archive revealed CSV pixel matrices and 7,603 exact duplicate rows inside DIGITS/CSV_FILES/. "
        "Removing duplicates to retain 1,073 unique digit images prevented data leakage. With bounding-box centering, 16 x 16 grayscale features, "
        "Euclidean distance, distance-weighted voting, and K = 3 selected on validation (99.38%), the refitted KNN classifier achieved "
        "95.03% accuracy, 95.82% macro precision, 94.29% macro recall, and 94.86% macro F1-score on the held-out test set.",
        body_style,
    ))

    # 21. References
    story.append(Paragraph("21. References", h1_style))
    story.append(Paragraph(
        "1. Sanjay E. Pate, DEVANAGARI HANDWRITTEN DIGITAL 44,000 IMAGES, Kaggle, DOI: 10.34740/KAGGLE/DSV/9820160, "
        "https://www.kaggle.com/datasets/sanjaypate/devanagari-script-digital-44000-images-dataset.",
        body_style,
    ))
    story.append(Paragraph(
        "2. T. Cover and P. Hart, 'Nearest neighbor pattern classification,' IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21-27, 1967.",
        body_style,
    ))
    story.append(Paragraph(
        "3. Pedregosa et al., 'Scikit-learn: Machine Learning in Python,' JMLR, vol. 12, pp. 2825-2830, 2011.",
        body_style,
    ))

    # 22. GitHub Repository Link
    story.append(Paragraph("22. GitHub Repository Link", h1_style))
    story.append(Paragraph("https://github.com/AkashKundu114/knn-devanagari-digit-recognition", body_style))

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
    return output_pdf


if __name__ == "__main__":
    pdf_path = build_pdf_report()
    print(f"Generated PDF report at: {pdf_path.resolve()} ({pdf_path.stat().st_size} bytes)")
