import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def get_project_root() -> Path:
    """Uses pathlib to dynamically find the project root folder."""
    current_path = Path(__file__).resolve().parent
    if current_path.name == 'src':
        return current_path.parent
    return current_path

def load_and_validate_data(root: Path) -> pd.DataFrame:
    """Loads labeled dataset and prints label verification details."""
    labeled_path = root / "data" / "processed" / "customer_support_tickets_labeled.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset missing at expected path: {labeled_path}")
        
    df = pd.read_csv(labeled_path)
    
    print(f"Total samples loaded: {len(df):,}")
    counts = df['derived_category'].value_counts()
    pcts = (df['derived_category'].value_counts(normalize=True) * 100).round(2)
    
    print("\n--- Derived Category Verification ---")
    print(f"Number of categories: {len(counts)}")
    for cat, cnt in counts.items():
        print(f"  - {cat:<27}: {cnt:,} ({pcts[cat]:.2f}%)")
        
    return df

def plot_and_save_confusion_matrix(cm: np.ndarray, classes: list, title: str, filepath: Path):
    """Plots and saves a normalized/raw confusion matrix heatmap."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel('Predicted Category', fontsize=11)
    plt.ylabel('Actual Category', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    root = get_project_root()
    models_dir = root / "models"
    reports_dir = root / "reports"
    docs_dir = reports_dir / "docs"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Project root identified: {root}")
    
    try:
        # Step 1 & 2: Load and validate labeled dataset
        df = load_and_validate_data(root)
        
        # Step 3: Train/Test Split (80/20 Stratified)
        X = df['ticket_text'].fillna('').astype(str)
        y = df['derived_category'].astype(str)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        print("\n--- Train/Test Split Distribution ---")
        print(f"Training samples: {len(X_train):,}")
        print(f"Testing samples : {len(X_test):,}")
        
        # Step 4: TF-IDF Feature Extraction (Fit ONLY on X_train)
        print("\n--- TF-IDF Feature Extraction (Fitted ONLY on X_train) ---")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        n_features = len(vectorizer.get_feature_names_out())
        print(f"TF-IDF Vocabulary Size: {n_features:,} features")
        print(f"X_train_tfidf shape    : {X_train_tfidf.shape}")
        print(f"X_test_tfidf shape     : {X_test_tfidf.shape}")
        
        # Step 5 & 6: Train and Evaluate 5 Model Variants
        candidate_models = {
            'Logistic Regression (Standard)': (
                LogisticRegression(random_state=42, max_iter=1000),
                'Standard',
                'derived_category_logistic_regression_standard_confusion_matrix.png'
            ),
            'Logistic Regression (Balanced)': (
                LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'Balanced',
                'derived_category_logistic_regression_balanced_confusion_matrix.png'
            ),
            'Linear SVM (Standard)': (
                LinearSVC(random_state=42, max_iter=2000),
                'Standard',
                'derived_category_linearsvc_standard_confusion_matrix.png'
            ),
            'Linear SVM (Balanced)': (
                LinearSVC(random_state=42, max_iter=2000, class_weight='balanced'),
                'Balanced',
                'derived_category_linearsvc_balanced_confusion_matrix.png'
            ),
            'Multinomial Naive Bayes': (
                MultinomialNB(),
                'None (N/A)',
                'derived_category_naive_bayes_confusion_matrix.png'
            )
        }
        
        classes = sorted(list(y.unique()))
        results = []
        model_predictions = {}
        fitted_model_objs = {}
        
        print("\n--- Training and Evaluating Candidates ---")
        for name, (model_obj, cw_label, fig_name) in candidate_models.items():
            print(f"Training {name}...")
            model_obj.fit(X_train_tfidf, y_train)
            preds = model_obj.predict(X_test_tfidf)
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
            w_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
            m_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            
            results.append({
                'Model': name,
                'Class Weight': cw_label,
                'Accuracy': round(acc, 4),
                'Weighted Precision': round(prec, 4),
                'Weighted Recall': round(rec, 4),
                'Weighted F1': round(w_f1, 4),
                'Macro F1': round(m_f1, 4)
            })
            
            model_predictions[name] = preds
            fitted_model_objs[name] = model_obj
            
            # Confusion matrix
            cm = confusion_matrix(y_test, preds, labels=classes)
            fig_path = figures_dir / fig_name
            plot_and_save_confusion_matrix(cm, classes, f"Confusion Matrix - {name}", fig_path)
            
        # Step 7: Model Comparison CSV
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Macro F1', 'Weighted F1'], ascending=[False, False])
        
        comp_csv = tables_dir / "category_derived_model_comparison.csv"
        results_df.to_csv(comp_csv, index=False)
        print(f"\nSaved model comparison table to: {comp_csv.relative_to(root)}")
        print("\n--- Model Comparison Table ---")
        print(results_df.to_string(index=False))
        
        # Select best model based on Macro F1
        best_row = results_df.iloc[0]
        best_model_name = best_row['Model']
        best_model_obj = fitted_model_objs[best_model_name]
        best_preds = model_predictions[best_model_name]
        
        print(f"\nBest Model Selected: {best_model_name} (Macro F1: {best_row['Macro F1']:.4f}, Weighted F1: {best_row['Weighted F1']:.4f})")
        
        # Step 8: Per-class performance
        cls_report_dict = classification_report(y_test, best_preds, output_dict=True, zero_division=0)
        per_class_df = pd.DataFrame(cls_report_dict).transpose()
        print("\n--- Per-Class Performance (Best Model) ---")
        print(per_class_df.loc[classes, ['precision', 'recall', 'f1-score']].to_string())
        
        # Step 9: Error Analysis & Misclassified Examples
        test_df = pd.DataFrame({
            'ticket_text': X_test,
            'actual_category': y_test,
            'predicted_category': best_preds
        })
        misclassified_df = test_df[test_df['actual_category'] != test_df['predicted_category']]
        
        mis_csv = tables_dir / "category_derived_misclassified_examples.csv"
        misclassified_df.to_csv(mis_csv, index=False)
        print(f"\nSaved misclassified test examples: {mis_csv.relative_to(root)} ({len(misclassified_df)} rows)")
        
        # Step 10: Save Best Model & Vectorizer
        model_pkl = models_dir / "derived_category_model.pkl"
        vec_pkl = models_dir / "derived_category_tfidf_vectorizer.pkl"
        
        joblib.dump(best_model_obj, model_pkl)
        joblib.dump(vectorizer, vec_pkl)
        
        print(f"Saved best model to : {model_pkl.relative_to(root)}")
        print(f"Saved vectorizer to : {vec_pkl.relative_to(root)}")
        
        # Step 11: Create DERIVED_CATEGORY_MODEL_REPORT.md
        report_path = docs_dir / "DERIVED_CATEGORY_MODEL_REPORT.md"
        
        report_content = f"""# Derived Category Classification Model Report (Stage 5D)

This report documents the machine learning training, evaluation, and model selection for predicting support-ticket categories (`derived_category`).

---

## ⚠️ ETHICAL DISCLAIMER & TRANSPARENCY NOTICE

**IMPORTANT NOTICE**:
The original `Ticket Type` labels were found to be non-predictive/noisy during diagnostic analysis (Stage 5A). Therefore, the model was trained using a separately derived category label created through documented rules (Stage 5B). **These derived labels are not equivalent to human-annotated ground truth.**

---

## 1. Executive Summary

* **Task**: Multi-class text classification across 7 support categories.
* **Dataset**: `customer_support_tickets_labeled.csv` (8,469 total records).
* **Train / Test Split**: 80% training (`6,775` samples), 20% testing (`1,694` samples), stratified (`random_state=42`).
* **Feature Extraction**: TF-IDF Unigrams + Bigrams (`10,000` max features), fitted **strictly on X_train**.
* **Winning Model**: **{best_model_name}**
* **Test Metrics**:
  * **Accuracy**: `{best_row['Accuracy']:.4f}` ({best_row['Accuracy']*100:.2f}%)
  * **Macro F1-Score**: **`{best_row['Macro F1']:.4f}`**
  * **Weighted F1-Score**: **`{best_row['Weighted F1']:.4f}`**

---

## 2. Derived Category Scheme & Distribution

| Derived Category | Train Count | Test Count | Total Count | Total % |
| :--- | :---: | :---: | :---: | :---: |
"""
        for cat in classes:
            tr_cnt = (y_train == cat).sum()
            te_cnt = (y_test == cat).sum()
            tot_cnt = tr_cnt + te_cnt
            tot_pct = (tot_cnt / len(df)) * 100
            report_content += f"| **{cat}** | {tr_cnt:,} | {te_cnt:,} | {tot_cnt:,} | {tot_pct:.2f}%\n"

        report_content += f"""
---

## 3. Model Comparison Table

All models were evaluated on the held-out test set (`1,694` samples):

| Model Name | Class Weighting | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for idx, row in results_df.iterrows():
            report_content += f"| **{row['Model']}** | {row['Class Weight']} | {row['Accuracy']:.4f} | {row['Weighted Precision']:.4f} | {row['Weighted Recall']:.4f} | {row['Weighted F1']:.4f} | **{row['Macro F1']:.4f}** |\n"

        report_content += f"""
---

## 4. Per-Class Performance ({best_model_name})

Evaluating detailed class-level performance across the 9.7:1 class imbalance:

| Derived Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
        for cat in classes:
            p_val = cls_report_dict[cat]['precision']
            r_val = cls_report_dict[cat]['recall']
            f1_val = cls_report_dict[cat]['f1-score']
            sup = cls_report_dict[cat]['support']
            report_content += f"| **{cat}** | {p_val:.4f} | {r_val:.4f} | **{f1_val:.4f}** | {sup:,} |\n"

        report_content += f"""
---

## 5. Error Analysis & Confusion Matrix Findings

* **Misclassified Test Samples**: `{len(misclassified_df)}` out of `{len(y_test):,}` test cases ({len(misclassified_df)/len(y_test)*100:.2f}% error rate).
* **Saved Misclassifications File**: [`category_derived_misclassified_examples.csv`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/category_derived_misclassified_examples.csv)
* **Confusion Matrix Visualization**: [`derived_category_linearsvc_balanced_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/derived_category_linearsvc_balanced_confusion_matrix.png)

---

## 6. Data Leakage Prevention Verification

1. **TF-IDF Vectorizer**: Fitted **strictly on X_train** (`6,775` samples). `X_test` was transformed using `vectorizer.transform(X_test)`.
2. **Feature Exclusion**: Excluded all post-resolution fields (`Resolution`, `Time to Resolution`, `Customer Satisfaction Rating`, `First Response Time`, `Ticket Status`).
3. **No Target Leakage**: The original noisy `Ticket Type` column was completely excluded from feature sets and model target variables.

---

## 7. Artifact Persistence & Next Steps

* **Saved Model**: [`models/derived_category_model.pkl`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/models/derived_category_model.pkl)
* **Saved Vectorizer**: [`models/derived_category_tfidf_vectorizer.pkl`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/models/derived_category_tfidf_vectorizer.pkl)
* **Next Stage**: Proceed to **Stage 6 (Ticket Priority Classification Model)**.
"""
        report_path.write_text(report_content, encoding='utf-8')
        print(f"Saved training report to     : {report_path.relative_to(root)}")
        
        # Step 13: Final Summary Output
        print("\n" + "=" * 45)
        print("STAGE 5D COMPLETED")
        print("=" * 45)
        print(f"Training samples: {len(X_train):,}")
        print(f"Testing samples : {len(X_test):,}")
        print(f"Number of categories: 7")
        print(f"TF-IDF features: {n_features:,}")
        print(f"\nBest model: {best_model_name}")
        print(f"Best Macro F1: {best_row['Macro F1']:.4f}")
        print(f"Best Weighted F1: {best_row['Weighted F1']:.4f}")
        print(f"Accuracy: {best_row['Accuracy']:.4f}")
        print(f"\nModel saved:\n  models/derived_category_model.pkl")
        print(f"\nVectorizer saved:\n  models/derived_category_tfidf_vectorizer.pkl")
        print("=" * 45 + "\n")
        
    except Exception as e:
        print(f"An error occurred during category model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
