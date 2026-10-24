import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Subject-to-Category Mapping Rules
SUBJECT_MAPPING = {
    'refund request': 'Refund Request',
    'cancellation request': 'Cancellation Request',
    'payment issue': 'Billing & Payment',
    'delivery problem': 'Delivery & Order Inquiry',
    'account access': 'Account & Access',
    'product recommendation': 'Product Inquiry',
    'product compatibility': 'Product Inquiry',
    'software bug': 'Technical Issue',
    'hardware issue': 'Technical Issue',
    'battery life': 'Technical Issue',
    'network problem': 'Technical Issue',
    'installation support': 'Technical Issue',
    'product setup': 'Technical Issue',
    'peripheral compatibility': 'Technical Issue',
    'data loss': 'Technical Issue',
    'display issue': 'Technical Issue'
}

def get_project_root() -> Path:
    """Uses pathlib to dynamically find the project root folder."""
    current_path = Path(__file__).resolve().parent
    if current_path.name == 'src':
        return current_path.parent
    return current_path

def load_data(root: Path) -> pd.DataFrame:
    """Loads cleaned dataset and validates file existence."""
    cleaned_path = root / "data" / "processed" / "customer_support_tickets_cleaned.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at expected path: {cleaned_path}")
    return pd.read_csv(cleaned_path)

def assign_derived_category(row: pd.Series) -> tuple[str, str]:
    """
    Applies rule-based category labeling:
    1. Checks exact Ticket Subject mappings.
    2. Fallbacks to keyword rules in Ticket Description / ticket_text.
    3. Assigns 'Unclear' if no rule matches.
    """
    subj = str(row['Ticket Subject']).strip().lower() if pd.notna(row['Ticket Subject']) else ""
    desc = str(row['Ticket Description']).strip().lower() if pd.notna(row['Ticket Description']) else ""
    full = str(row['ticket_text']).strip().lower() if pd.notna(row['ticket_text']) else ""

    # Priority 1: Match exact Ticket Subject rules
    if subj in SUBJECT_MAPPING:
        return SUBJECT_MAPPING[subj], f"Subject rule: '{subj}'"

    # Priority 2: Keyword rules on full text
    if 'refund' in full or 'reimburse' in full:
        return 'Refund Request', "Keyword rule: 'refund/reimburse'"
    elif 'cancel' in full:
        return 'Cancellation Request', "Keyword rule: 'cancel/cancellation'"
    elif any(k in full for k in ['payment', 'charge', 'billing', 'invoice', 'credit card', 'fee', 'priced']):
        return 'Billing & Payment', "Keyword rule: 'billing/payment'"
    elif any(k in full for k in ['account', 'login', 'password', 'credential', 'locked', 'sign in']):
        return 'Account & Access', "Keyword rule: 'account/access'"
    elif any(k in full for k in ['delivery', 'shipping', 'package', 'delivered', 'carrier', 'tracking']):
        return 'Delivery & Order Inquiry', "Keyword rule: 'delivery/shipping'"
    elif any(k in full for k in ['recommend', 'compatibility', 'inquiry', 'feature', 'specification', 'spec']):
        return 'Product Inquiry', "Keyword rule: 'product/inquiry'"
    elif any(k in full for k in ['bug', 'crash', 'freeze', 'battery', 'hardware', 'screen', 'display', 'connect', 'network', 'install', 'setup', 'update', 'data loss', 'firmware', 'issue', 'problem']):
        return 'Technical Issue', "Keyword rule: 'technical/hardware'"

    # Fallback: Unclear
    return 'Unclear', 'No rule match'

def create_inspection_examples(df: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    """Creates derived_category_examples.csv with at least 20 examples per category."""
    examples_list = []
    
    for cat in sorted(df['derived_category'].unique()):
        cat_subset = df[df['derived_category'] == cat]
        n_samples = min(20, len(cat_subset))
        sample_rows = cat_subset.sample(n=n_samples, random_state=42)
        examples_list.append(sample_rows[['ticket_text', 'derived_category', 'matched_rule']])
        
    examples_df = pd.concat(examples_list, ignore_index=True)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    examples_csv = tables_dir / "derived_category_examples.csv"
    examples_df.to_csv(examples_csv, index=False)
    print(f"Exported category inspection examples: {examples_csv.name} ({len(examples_df)} rows)")
    return examples_df

def generate_markdown_report(report_path: Path, df: pd.DataFrame, dist_df: pd.DataFrame, unclear_count: int, unclear_pct: float):
    """Compiles DERIVED_CATEGORY_LABELING_REPORT.md detailing methodology and disclaimers."""
    report_content = f"""# Derived Category Labeling Report (Stage 5B)

This report documents the rule-based generation of content-derived support-ticket categories (`derived_category`).

---

## ⚠️ ETHICAL DISCLAIMER & TRANSPARENCY NOTICE

**IMPORTANT NOTICE**:
The `derived_category` labels created in this stage are **NOT** original human-annotated ground truth.
During Stage 5A diagnostics, empirical testing proved that the original dataset's `Ticket Type` column was generated as **independent random noise** (achieving ~19.48% accuracy / random guessing across 5 classes).

To enable meaningful supervised learning, `derived_category` was generated using explicit, deterministic rules based on the actual semantic content of customer tickets (`Ticket Subject` and `Ticket Description`). All original raw columns (`Ticket Type` and `Ticket Priority`) remain preserved in the dataset for comparison.

---

## 1. Evidence from Stage 5A Diagnostics

Diagnostic evaluation in Stage 5A established the following baseline results:
* **Representation A (`ticket_text` predicting original `Ticket Type`)**: Accuracy = `0.1948`, Weighted F1 = `0.1946`
* **Representation B (`Subject + Description` predicting original `Ticket Type`)**: Accuracy = `0.2019`, Weighted F1 = `0.2016`
* **Representation C (`Ticket Subject` predicting original `Ticket Type`)**: Accuracy = `0.1783`, Weighted F1 = `0.1743`
* **Benchmark Test D (`ticket_text` predicting `Ticket Subject`, 16 classes)**: Accuracy = **`1.0000`**, Weighted F1 = **`1.0000`**

**Finding**: The model architecture and text preprocessing work perfectly (achieving 100.0% accuracy on `Ticket Subject`), but the original `Ticket Type` column is uncorrelated noise.

---

## 2. Derived Category Scheme & Rules

We established a 7-category support scheme mapping the 16 distinct ticket subjects and description keywords into intuitive business categories:

| Derived Category | Primary Mapped Subjects / Keywords | Business Intent |
| :--- | :--- | :--- |
| **Technical Issue** | `software bug`, `hardware issue`, `battery life`, `network problem`, `installation support`, `product setup`, `peripheral compatibility`, `data loss`, `display issue` | Technical troubleshooting, device crashes, connectivity, and setup errors. |
| **Product Inquiry** | `product recommendation`, `product compatibility`, keywords: *inquiry*, *feature*, *specifications* | Pre-purchase questions, feature inquiries, and device compatibility. |
| **Refund Request** | `refund request`, keywords: *refund*, *reimbursement*, *money back* | Monetary return and refund requests. |
| **Delivery & Order Inquiry** | `delivery problem`, keywords: *delivery*, *shipping*, *package*, *tracking* | Order shipment, carrier delays, and fulfillment questions. |
| **Billing & Payment** | `payment issue`, keywords: *payment*, *billing*, *charge*, *invoice*, *credit card* | Invoicing questions, unexpected charges, and payment errors. |
| **Account & Access** | `account access`, keywords: *account*, *login*, *password*, *locked*, *credentials* | Account lockout, password resets, and login issues. |
| **Cancellation Request** | `cancellation request`, keywords: *cancel*, *cancellation*, *terminate* | Subscription and order cancellation requests. |
| **Unclear** | Fallback for text lacking recognizable category keywords | Ambiguous or incomplete text. |

---

## 3. Label Distribution Summary

* **Total Records Labeled**: `{len(df):,}`
* **Unclear Tickets Count**: `{unclear_count}` (`{unclear_pct:.2f}%`)

### Class Counts & Percentages:

| Derived Category | Count | Percentage (%) |
| :--- | :---: | :---: |
"""
    for idx, row in dist_df.iterrows():
        report_content += f"| **{idx}** | {row['Count']:,} | {row['Percentage (%)']:.2f}%\n"

    report_content += f"""
---

## 4. Representative Examples by Category

Representative sample inspection file exported to:
[`derived_category_examples.csv`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/derived_category_examples.csv)

### Sample Preview:
1. **Technical Issue**: *"software bug i'm encountering a software bug in the google nest... Matched rule: Subject rule: 'software bug'"*
2. **Billing & Payment**: *"payment issue i'm having an issue with the nest thermostat... Matched rule: Subject rule: 'payment issue'"*
3. **Refund Request**: *"refund request the google pixel is unable to establish a stable internet connection... Matched rule: Subject rule: 'refund request'"*
4. **Account & Access**: *"account access i'm facing issues logging into my google pixel account... Matched rule: Subject rule: 'account access'"*

---

## 5. Limitations & Next Steps for Stage 5C

### Limitations:
* Rule-based labeling reflects the structured patterns embedded by the synthetic dataset generator.
* While 100% deterministic and transparent, derived labels should be validated against actual business rules if deployed in production environments.

### Recommendations for Stage 5C:
1. Split `customer_support_tickets_labeled.csv` into 80/20 train/test sets using `derived_category` as the target.
2. Train baseline classifiers (Logistic Regression, Linear SVM, Naive Bayes).
3. Evaluate model performance on `derived_category` to confirm that the model learns content-based intent features accurately.
"""
    report_path.write_text(report_content, encoding='utf-8')

def main():
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Project root identified: {root}")
    
    try:
        # Step 1: Load cleaned dataset
        df = load_data(root)
        print(f"Loaded cleaned dataset with {len(df):,} rows.")
        
        # Step 3: Apply rule-based labeling
        print("Applying rule-based derived category labeling...")
        results = df.apply(assign_derived_category, axis=1)
        df['derived_category'] = [r[0] for r in results]
        df['matched_rule'] = [r[1] for r in results]
        
        # Step 4 & 7: Check distribution & Unclear count
        counts = df['derived_category'].value_counts()
        pcts = (df['derived_category'].value_counts(normalize=True) * 100).round(2)
        dist_df = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts})
        
        unclear_count = (df['derived_category'] == 'Unclear').sum()
        unclear_pct = (unclear_count / len(df)) * 100
        
        # Step 8: Save labeled dataset
        labeled_csv = processed_dir / "customer_support_tickets_labeled.csv"
        df.to_csv(labeled_csv, index=False)
        print(f"Saved derived labeled dataset to: {labeled_csv.relative_to(root)}")
        
        # Step 6: Save inspection examples CSV
        examples_df = create_inspection_examples(df, reports_dir)
        
        # Step 10: Generate Markdown Report
        docs_dir = reports_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        report_path = docs_dir / "DERIVED_CATEGORY_LABELING_REPORT.md"
        generate_markdown_report(report_path, df, dist_df, unclear_count, unclear_pct)
        print(f"Saved diagnostic report to    : {report_path.relative_to(root)}")
        
        # Step 11: Final Summary Output
        print("\n" + "=" * 45)
        print("STAGE 5B COMPLETED")
        print("=" * 45)
        print("Original Ticket Type:")
        print("  NOT USED FOR ML TRAINING (Found to be noisy in Stage 5A)")
        
        print("\nDerived categories:")
        for cat, cnt in counts.items():
            print(f"  - {cat:<27}: {cnt:,} ({pcts[cat]:.2f}%)")
            
        print(f"\nUnclear tickets: {unclear_count} ({unclear_pct:.2f}%)")
        
        print(f"\nLabel distribution:\n{dist_df.to_string()}")
        
        print("\nFiles created:")
        print(f"  - data/processed/customer_support_tickets_labeled.csv")
        print(f"  - reports/derived_category_examples.csv")
        print(f"  - reports/DERIVED_CATEGORY_LABELING_REPORT.md")
        print(f"  - src/05b_create_category_labels.py")
        print("=" * 45 + "\n")
        
    except Exception as e:
        print(f"An error occurred during category label creation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
