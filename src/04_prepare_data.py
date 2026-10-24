import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_project_root() -> Path:
    """
    Uses pathlib to dynamically find the project root folder.
    Handles running from either project root or src/ directory.
    """
    current_path = Path(__file__).resolve().parent
    if current_path.name == 'src':
        return current_path.parent
    return current_path

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Loads cleaned dataset and validates file existence."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at expected path: {file_path}")
    return pd.read_csv(file_path)

def validate_columns(df: pd.DataFrame, required_cols: list) -> None:
    """Verifies that all required columns exist in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns for ML preparation: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

def clean_and_filter_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans text and targets, and removes invalid rows where text or target is missing/empty.
    Tracks and returns removal statistics.
    """
    initial_count = len(df)
    removal_stats = {
        'initial_rows': initial_count,
        'missing_text_dropped': 0,
        'empty_text_dropped': 0,
        'missing_category_dropped': 0,
        'missing_priority_dropped': 0,
        'final_rows': 0
    }

    df_clean = df.copy()

    # Step 1: Target label cleaning (strip whitespace)
    df_clean['Ticket Type'] = df_clean['Ticket Type'].astype(str).str.strip()
    df_clean['Ticket Priority'] = df_clean['Ticket Priority'].astype(str).str.strip()

    # Check missing targets
    missing_cat_mask = df_clean['Ticket Type'].isna() | (df_clean['Ticket Type'] == '') | (df_clean['Ticket Type'] == 'nan')
    missing_prio_mask = df_clean['Ticket Priority'].isna() | (df_clean['Ticket Priority'] == '') | (df_clean['Ticket Priority'] == 'nan')

    removal_stats['missing_category_dropped'] = missing_cat_mask.sum()
    removal_stats['missing_priority_dropped'] = missing_prio_mask.sum()

    df_clean = df_clean[~missing_cat_mask & ~missing_prio_mask]

    # Step 2: Clean ticket_text (ensure string, strip, normalize whitespace)
    # Note: Negations (not, no, never, cannot, failed) are preserved naturally
    df_clean['ticket_text'] = df_clean['ticket_text'].fillna('').astype(str).str.strip()
    df_clean['ticket_text'] = df_clean['ticket_text'].str.replace(r'\s+', ' ', regex=True)

    # Filter out empty text
    empty_text_mask = df_clean['ticket_text'] == ''
    removal_stats['empty_text_dropped'] = empty_text_mask.sum()
    df_clean = df_clean[~empty_text_mask]

    removal_stats['final_rows'] = len(df_clean)
    removal_stats['total_removed'] = initial_count - len(df_clean)

    return df_clean, removal_stats

def validate_target_classes(df: pd.DataFrame) -> None:
    """Validates target categories and priorities against expected values."""
    expected_categories = {'Refund request', 'Technical issue', 'Cancellation request', 'Product inquiry', 'Billing inquiry'}
    expected_priorities = {'Low', 'Medium', 'High', 'Critical'}

    actual_categories = set(df['Ticket Type'].unique())
    actual_priorities = set(df['Ticket Priority'].unique())

    unexpected_cat = actual_categories - expected_categories
    unexpected_prio = actual_priorities - expected_priorities

    if unexpected_cat:
        print(f"[WARNING] Unexpected categories found in dataset: {unexpected_cat}")
    if unexpected_prio:
        print(f"[WARNING] Unexpected priorities found in dataset: {unexpected_prio}")

def perform_stratified_split(df: pd.DataFrame, target_col: str, test_size=0.20, random_state=42):
    """
    Performs stratified train/test split for a given target column.
    Returns train_df, test_df.
    """
    X = df[['ticket_text']]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = pd.DataFrame({'ticket_text': X_train['ticket_text'], target_col: y_train})
    test_df = pd.DataFrame({'ticket_text': X_test['ticket_text'], target_col: y_test})

    return train_df, test_df

def compute_distribution_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Computes comparative class distribution summary between train and test splits."""
    train_counts = train_df[target_col].value_counts()
    train_pcts = train_df[target_col].value_counts(normalize=True) * 100

    test_counts = test_df[target_col].value_counts()
    test_pcts = test_df[target_col].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        'Train Count': train_counts,
        'Train %': train_pcts.round(2),
        'Test Count': test_counts,
        'Test %': test_pcts.round(2)
    })
    return summary

def generate_markdown_report(report_path: Path, stats: dict, cat_dist: pd.DataFrame, prio_dist: pd.DataFrame) -> None:
    """Generates ML_DATA_PREPARATION_REPORT.md detailing the preparation process."""
    report_content = f"""# ML Data Preparation Report

This report documents the machine learning data preparation phase for the Support Ticket Classification project.

---

## 1. Dataset Row Counts & Removal Summary

* **Starting Row Count**: `{stats['initial_rows']:,}`
* **Final Prepared Rows**: `{stats['final_rows']:,}`
* **Total Rows Removed**: `{stats['total_removed']}`
  * Missing/Empty Category Target: `{stats['missing_category_dropped']}`
  * Missing/Empty Priority Target: `{stats['missing_priority_dropped']}`
  * Missing/Empty Ticket Text: `{stats['empty_text_dropped']}`

---

## 2. Feature & Target Definitions

### ML Task A: Category Classification
* **Feature Input (`X`)**: `ticket_text` (Combined Subject + Description text)
* **Target Label (`y_category`)**: `Ticket Type` (5 classes)

### ML Task B: Priority Classification
* **Feature Input (`X`)**: `ticket_text` (Combined Subject + Description text)
* **Target Label (`y_priority`)**: `Ticket Priority` (4 classes)

---

## 3. Data Leakage Control & Excluded Columns

The following post-resolution columns were **strictly excluded** from ML prediction features to prevent data leakage:
* `Resolution`
* `Time to Resolution`
* `Customer Satisfaction Rating`
* `First Response Time`
* `Ticket Status`

---

## 4. Train / Test Split Parameters

* **Test Set Percentage**: `20.0%` (`test_size = 0.20`)
* **Training Set Percentage**: `80.0%`
* **Random State Seed**: `42` (ensures exact reproducibility across runs)
* **Stratification Method**: Stratified random sampling based on target class labels (`stratify = y`)

---

## 5. Target A: Ticket Category Class Distribution

* **Category Training Rows**: `{stats['cat_train_rows']:,}`
* **Category Testing Rows**: `{stats['cat_test_rows']:,}`

### Train vs. Test Class Comparison:

| Category Class | Train Count | Train (%) | Test Count | Test (%) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for idx, row in cat_dist.iterrows():
        report_content += f"| **{idx}** | {row['Train Count']:,} | {row['Train %']:.2f}% | {row['Test Count']:,} | {row['Test %']:.2f}%\n"

    report_content += f"""
---

## 6. Target B: Ticket Priority Class Distribution

* **Priority Training Rows**: `{stats['prio_train_rows']:,}`
* **Priority Testing Rows**: `{stats['prio_test_rows']:,}`

### Train vs. Test Class Comparison:

| Priority Class | Train Count | Train (%) | Test Count | Test (%) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for idx, row in prio_dist.iterrows():
        report_content += f"| **{idx}** | {row['Train Count']:,} | {row['Train %']:.2f}% | {row['Test Count']:,} | {row['Test %']:.2f}%\n"

    report_content += """
---

## 7. Vectorization & Data Leakage Protection Policy

**IMPORTANT**: TF-IDF Vectorization was **NOT** fitted during this stage.

* **Rationale**: Fitting TF-IDF on the combined dataset (or test set) would allow vocabulary frequencies and IDF statistics from the test set to leak into the training process.
* **Stage 5 Protocol**:
  1. Fit `TfidfVectorizer` **ONLY** on `category_train.csv` / `priority_train.csv`.
  2. Transform training text using the fitted vectorizer.
  3. Transform test text using the already-fitted vectorizer (with zero refitting).

---

## 8. Output Datasets & Verification

The following prepared datasets have been exported to `data/processed/`:
1. `category_train.csv` (`ticket_text`, `Ticket Type`)
2. `category_test.csv` (`ticket_text`, `Ticket Type`)
3. `priority_train.csv` (`ticket_text`, `Ticket Priority`)
4. `priority_test.csv` (`ticket_text`, `Ticket Priority`)

**Status: DATA IS 100% READY FOR SUPERVISED MODEL TRAINING (STAGE 5).**
"""
    report_path.write_text(report_content, encoding='utf-8')

def main():
    root = get_project_root()
    cleaned_csv = root / "data" / "processed" / "customer_support_tickets_cleaned.csv"
    output_dir = root / "data" / "processed"
    report_file = root / "reports" / "ML_DATA_PREPARATION_REPORT.md"

    print(f"Project root identified: {root}")
    print(f"Loading cleaned dataset from: {cleaned_csv}")

    try:
        # Step 1: Load Data
        df = load_dataset(cleaned_csv)

        # Step 2: Validate columns
        required_cols = ['ticket_text', 'Ticket Type', 'Ticket Priority']
        validate_columns(df, required_cols)

        # Step 3, 4, 5: Clean ML text, clean target labels, remove invalid rows
        df_clean, removal_stats = clean_and_filter_data(df)
        validate_target_classes(df_clean)

        # Step 6 & 8: Perform Stratified Splits for Task A & Task B
        cat_train, cat_test = perform_stratified_split(df_clean, 'Ticket Type', test_size=0.20, random_state=42)
        prio_train, prio_test = perform_stratified_split(df_clean, 'Ticket Priority', test_size=0.20, random_state=42)

        removal_stats['cat_train_rows'] = len(cat_train)
        removal_stats['cat_test_rows'] = len(cat_test)
        removal_stats['prio_train_rows'] = len(prio_train)
        removal_stats['prio_test_rows'] = len(prio_test)

        # Save prepared CSV files
        cat_train_path = output_dir / "category_train.csv"
        cat_test_path = output_dir / "category_test.csv"
        prio_train_path = output_dir / "priority_train.csv"
        prio_test_path = output_dir / "priority_test.csv"

        cat_train.to_csv(cat_train_path, index=False)
        cat_test.to_csv(cat_test_path, index=False)
        prio_train.to_csv(prio_train_path, index=False)
        prio_test.to_csv(prio_test_path, index=False)

        # Step 10: Compute Class Distribution Summaries
        cat_dist = compute_distribution_summary(cat_train, cat_test, 'Ticket Type')
        prio_dist = compute_distribution_summary(prio_train, prio_test, 'Ticket Priority')

        # Step 9: Generate Markdown Report
        generate_markdown_report(report_file, removal_stats, cat_dist, prio_dist)

        # Step 12: Final Summary Output
        print("\n" + "=" * 45)
        print("ML DATA PREPARATION COMPLETED")
        print("=" * 45)
        print(f"Starting rows: {removal_stats['initial_rows']:,}")
        print(f"Final rows: {removal_stats['final_rows']:,}")
        print(f"\nCategory training rows: {len(cat_train):,}")
        print(f"Category testing rows: {len(cat_test):,}")
        print(f"\nPriority training rows: {len(prio_train):,}")
        print(f"Priority testing rows: {len(prio_test):,}")
        
        print(f"\nCategory classes: {list(cat_train['Ticket Type'].unique())}")
        print(f"Priority classes: {list(prio_train['Ticket Priority'].unique())}")
        
        print("\nData leakage protection:")
        print("PASSED (TF-IDF vectorization deferred to Stage 5; post-resolution fields excluded)")
        
        print("\nOutput files:")
        print(f"  - {cat_train_path.relative_to(root)}")
        print(f"  - {cat_test_path.relative_to(root)}")
        print(f"  - {prio_train_path.relative_to(root)}")
        print(f"  - {prio_test_path.relative_to(root)}")
        print(f"  - {report_file.relative_to(root)}")
        print("=" * 45 + "\n")

    except Exception as e:
        print(f"An error occurred during ML data preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
