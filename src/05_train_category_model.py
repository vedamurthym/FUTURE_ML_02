import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def get_project_root() -> Path:
    """Uses pathlib to dynamically find the project root folder."""
    current_path = Path(__file__).resolve().parent
    if current_path.name == 'src':
        return current_path.parent
    return current_path

def load_data(root: Path):
    """Loads train and test CSV files for the category classification task."""
    train_path = root / "data" / "processed" / "category_train.csv"
    test_path = root / "data" / "processed" / "category_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Prepared category datasets not found. Paths checked: {train_path}, {test_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df['ticket_text'].fillna('').astype(str)
    y_train = train_df['Ticket Type'].astype(str)
    
    X_test = test_df['ticket_text'].fillna('').astype(str)
    y_test = test_df['Ticket Type'].astype(str)
    
    return X_train, y_train, X_test, y_test, test_df

def extract_tfidf_features(X_train, X_test):
    """
    Fits TF-IDF vectorizer strictly on X_train, then transforms X_test.
    Prevents data leakage from test text into feature representation.
    """
    print("\n--- Step 2: TF-IDF Feature Extraction ---")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True
    )
    
    # Fit strictly on X_train
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Transform X_test without fitting
    X_test_tfidf = vectorizer.transform(X_test)
    
    n_features = len(vectorizer.get_feature_names_out())
    print(f"Number of TF-IDF features extracted: {n_features:,}")
    print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
    print(f"X_test_tfidf shape : {X_test_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_and_evaluate_models(X_train_tfidf, y_train, X_test_tfidf, y_test, figures_dir: Path):
    """Trains Logistic Regression, LinearSVC, and MultinomialNB, evaluating each."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Linear SVM': LinearSVC(random_state=42, max_iter=2000),
        'Multinomial Naive Bayes': MultinomialNB()
    }
    
    fig_filenames = {
        'Logistic Regression': 'category_logistic_regression_confusion_matrix.png',
        'Linear SVM': 'category_linear_svm_confusion_matrix.png',
        'Multinomial Naive Bayes': 'category_naive_bayes_confusion_matrix.png'
    }
    
    results = []
    trained_models = {}
    predictions = {}
    
    classes = sorted(list(y_train.unique()))
    
    print("\n--- Step 3 & 4: Model Training and Evaluation ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4)
        })
        
        trained_models[name] = model
        predictions[name] = y_pred
        
        # Plot and save confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    annot_kws={"size": 11, "weight": "bold"})
        plt.title(f"Category Confusion Matrix: {name}", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Predicted Ticket Type", fontsize=11)
        plt.ylabel("Actual Ticket Type", fontsize=11)
        plt.xticks(rotation=20)
        plt.tight_layout()
        
        fig_path = figures_dir / fig_filenames[name]
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot: {fig_path.name}")
        
    results_df = pd.DataFrame(results)
    return results_df, trained_models, predictions

def perform_error_analysis(test_df: pd.DataFrame, y_test, best_y_pred, reports_dir: Path):
    """Identifies misclassified examples for the best model and exports them."""
    misclassified_mask = y_test != best_y_pred
    misclassified_df = pd.DataFrame({
        'ticket_text': test_df.loc[misclassified_mask, 'ticket_text'],
        'actual Ticket Type': y_test[misclassified_mask],
        'predicted Ticket Type': best_y_pred[misclassified_mask]
    })
    
    misclass_csv = reports_dir / "category_misclassified_examples.csv"
    misclassified_df.to_csv(misclass_csv, index=False)
    print(f"\nSaved misclassified examples to: {misclass_csv.name} ({len(misclassified_df)} examples)")
    return misclassified_df

def generate_markdown_report(report_path: Path, train_len, test_len, n_features, results_df: pd.DataFrame, best_name, best_f1, misclassified_df: pd.DataFrame):
    """Generates CATEGORY_MODEL_REPORT.md detailing model metrics and findings."""
    best_row = results_df[results_df['Model'] == best_name].iloc[0]
    
    # Calculate top confused pair
    top_confusion_desc = "None"
    if not misclassified_df.empty:
        confusion_counts = misclassified_df.groupby(['actual Ticket Type', 'predicted Ticket Type']).size().sort_values(ascending=False)
        if not confusion_counts.empty:
            top_pair = confusion_counts.index[0]
            top_count = confusion_counts.iloc[0]
            top_confusion_desc = f"'{top_pair[0]}' misclassified as '{top_pair[1]}' ({top_count} times)"

    report_content = f"""# Category Model Training & Evaluation Report

This report documents the baseline machine learning models developed to classify support tickets into **Ticket Type** categories.

---

## 1. Executive Summary & Objective

* **Objective**: Predict `Ticket Type` (5 categories) from incoming customer support ticket text (`ticket_text`).
* **Training Dataset Size**: `{train_len:,}` samples (80%)
* **Testing Dataset Size**: `{test_len:,}` samples (20%)
* **Best Model Selected**: **{best_name}**
* **Best Model F1-Score (Weighted)**: `{best_f1:.4f}`

---

## 2. Feature Extraction & TF-IDF Configuration

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts raw customer ticket text into numerical feature vectors based on how important words are within the corpus:
* **Term Frequency (TF)**: Measures how frequently a word appears in a specific ticket.
* **Inverse Document Frequency (IDF)**: Downweights words that appear commonly across all tickets (e.g. general terms) and highlights distinctive words.

### Vectorizer Configuration:
* **NGram Range**: `(1, 2)` (Includes both single words and two-word phrases like *"not working"*, *"billing charge"*)
* **Min Document Frequency (`min_df`)**: `2` (Ignores rare terms appearing in only 1 document)
* **Sublinear TF Scaling**: Enabled (applies `1 + log(tf)` to prevent dominant term counts)
* **Total Features Extracted**: `{n_features:,}` numerical TF-IDF features

---

## 3. Models Tested

1. **Logistic Regression**: Linear multiclass classifier (`multi_class='multinomial'`) optimizing log-loss.
2. **Linear Support Vector Machine (LinearSVC)**: Maximum-margin hyperplane classifier well-suited for high-dimensional text data.
3. **Multinomial Naive Bayes**: Probabilistic classifier applying Bayes' theorem with independence assumptions.

---

## 4. Evaluation Metrics & Comparison Table

Models were evaluated on the `{test_len:,}` held-out test tickets. Metrics use **weighted averaging** across all 5 category classes:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
"""
    for idx, row in results_df.iterrows():
        report_content += f"| **{row['Model']}** | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1 Score']:.4f} |\n"

    report_content += f"""
* **Primary Selection Metric**: **F1 Score** was chosen as the primary metric because it balances Precision and Recall across all multiclass categories.
* **Winning Model**: **{best_name}** achieved the highest F1-score of `{best_f1:.4f}`.

---

## 5. Confusion Matrix Observations & Error Analysis

### Confusion Matrix Plots:
Saved visualizations in `reports/figures/`:
- [`category_logistic_regression_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_logistic_regression_confusion_matrix.png)
- [`category_linear_svm_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_linear_svm_confusion_matrix.png)
- [`category_naive_bayes_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_naive_bayes_confusion_matrix.png)

### Error Analysis & Most Common Misclassification:
* **Misclassified Test Samples**: `{len(misclassified_df)}` out of `{test_len}` test tickets ({len(misclassified_df)/test_len*100:.2f}% error rate).
* **Most Frequent Confusion Pair**: {top_confusion_desc}.
* **Root Cause Analysis**: Synthetic ticket descriptions in this dataset share overlapping boilerplate phrasing (e.g. *"I'm having an issue with the product_purchased. Please assist."*) across different categories. Misclassifications happen when the ticket text lacks category-unique key terms.

---

## 6. Data Leakage Prevention Verification

* **TF-IDF Fitting Boundary**: The vectorizer was **fitted exclusively on `category_train.csv`**. The test dataset (`category_test.csv`) was transformed using the already-fitted vectorizer without any refitting.
* **Feature Boundaries**: Excluded all post-resolution fields (`Resolution`, `Time to Resolution`, `CSAT Rating`, `First Response Time`).

---

## 7. Model Persistence

The trained artifacts have been persisted:
* **Selected Model File**: [`models/category_model.pkl`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/models/category_model.pkl)
* **Fitted Vectorizer File**: [`models/category_tfidf_vectorizer.pkl`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/models/category_tfidf_vectorizer.pkl)

---

## 8. Conclusion

The **{best_name}** model is saved and ready for deployment or downstream inference tasks.
"""
    report_path.write_text(report_content, encoding='utf-8')

def main():
    root = get_project_root()
    figures_dir = root / "reports" / "figures"
    reports_dir = root / "reports"
    models_dir = root / "models"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Project root identified: {root}")
    
    try:
        # Step 1: Load train & test data
        X_train, y_train, X_test, y_test, test_df = load_data(root)
        print(f"Loaded training samples: {len(X_train):,}")
        print(f"Loaded testing samples : {len(X_test):,}")
        
        # Step 2: TF-IDF feature extraction
        vectorizer, X_train_tfidf, X_test_tfidf = extract_tfidf_features(X_train, X_test)
        
        # Step 3 & 4: Train & Evaluate 3 models
        results_df, trained_models, predictions = train_and_evaluate_models(
            X_train_tfidf, y_train, X_test_tfidf, y_test, figures_dir
        )
        
        # Step 5: Save comparison CSV & determine best model
        comparison_csv = reports_dir / "category_model_comparison.csv"
        results_df.to_csv(comparison_csv, index=False)
        print(f"\nSaved model comparison table to: {comparison_csv.name}")
        print("\n--- Model Comparison ---")
        print(results_df.to_string(index=False))
        
        # Select best model based on F1 Score
        best_row = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]
        best_name = best_row['Model']
        best_f1 = best_row['F1 Score']
        best_model = trained_models[best_name]
        best_y_pred = predictions[best_name]
        
        print(f"\nBest Model Selected: {best_name} (F1 Score: {best_f1:.4f})")
        
        # Step 6: Error Analysis
        misclassified_df = perform_error_analysis(test_df, y_test, best_y_pred, reports_dir)
        
        # Step 7: Save best model & vectorizer using joblib
        model_pkl = models_dir / "category_model.pkl"
        vectorizer_pkl = models_dir / "category_tfidf_vectorizer.pkl"
        
        joblib.dump(best_model, model_pkl)
        joblib.dump(vectorizer, vectorizer_pkl)
        
        print(f"Saved best model to: {model_pkl.relative_to(root)}")
        print(f"Saved vectorizer to : {vectorizer_pkl.relative_to(root)}")
        
        # Step 8: Generate Report
        report_path = reports_dir / "CATEGORY_MODEL_REPORT.md"
        n_features = X_train_tfidf.shape[1]
        generate_markdown_report(report_path, len(X_train), len(X_test), n_features, results_df, best_name, best_f1, misclassified_df)
        print(f"Saved report to    : {report_path.relative_to(root)}")
        
        # Step 10: Final Console Summary
        print("\n" + "=" * 45)
        print("CATEGORY MODEL TRAINING COMPLETED")
        print("=" * 45)
        print(f"Training samples: {len(X_train):,}")
        print(f"Testing samples: {len(X_test):,}")
        print(f"TF-IDF features: {n_features:,}")
        print(f"\nModel results:")
        for idx, row in results_df.iterrows():
            print(f"  - {row['Model']:<25}: Acc = {row['Accuracy']:.4f}, F1 = {row['F1 Score']:.4f}")
        print(f"\nBest model: {best_name}")
        print(f"Best F1-score: {best_f1:.4f}")
        print(f"\nSaved model:\n  models/category_model.pkl")
        print(f"\nSaved vectorizer:\n  models/category_tfidf_vectorizer.pkl")
        print("=" * 45 + "\n")
        
    except Exception as e:
        print(f"An error occurred during category model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
