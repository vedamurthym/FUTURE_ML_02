# Derived Category Classification Model Report (Stage 5D)

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
* **Winning Model**: **Linear SVM (Balanced)**
* **Test Metrics**:
  * **Accuracy**: `0.9982` (99.82%)
  * **Macro F1-Score**: **`0.9979`**
  * **Weighted F1-Score**: **`0.9982`**

---

## 2. Derived Category Scheme & Distribution

| Derived Category | Train Count | Test Count | Total Count | Total % |
| :--- | :---: | :---: | :---: | :---: |
| **Account & Access** | 407 | 102 | 509 | 6.01%
| **Billing & Payment** | 421 | 105 | 526 | 6.21%
| **Cancellation Request** | 389 | 98 | 487 | 5.75%
| **Delivery & Order Inquiry** | 449 | 112 | 561 | 6.62%
| **Product Inquiry** | 867 | 217 | 1,084 | 12.80%
| **Refund Request** | 461 | 115 | 576 | 6.80%
| **Technical Issue** | 3,781 | 945 | 4,726 | 55.80%

---

## 3. Model Comparison Table

All models were evaluated on the held-out test set (`1,694` samples):

| Model Name | Class Weighting | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Balanced)** | Balanced | 0.9982 | 0.9983 | 0.9982 | 0.9982 | **0.9979** |
| **Linear SVM (Standard)** | Standard | 0.9982 | 0.9982 | 0.9982 | 0.9982 | **0.9978** |
| **Logistic Regression (Standard)** | Standard | 0.9953 | 0.9953 | 0.9953 | 0.9953 | **0.9947** |
| **Logistic Regression (Balanced)** | Balanced | 0.9947 | 0.9948 | 0.9947 | 0.9947 | **0.9945** |
| **Multinomial Naive Bayes** | None (N/A) | 0.5579 | 0.3112 | 0.5579 | 0.3995 | **0.1023** |

---

## 4. Per-Class Performance (Linear SVM (Balanced))

Evaluating detailed class-level performance across the 9.7:1 class imbalance:

| Derived Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Account & Access** | 1.0000 | 1.0000 | **1.0000** | 102.0 |
| **Billing & Payment** | 1.0000 | 1.0000 | **1.0000** | 105.0 |
| **Cancellation Request** | 1.0000 | 1.0000 | **1.0000** | 98.0 |
| **Delivery & Order Inquiry** | 0.9912 | 1.0000 | **0.9956** | 112.0 |
| **Product Inquiry** | 1.0000 | 1.0000 | **1.0000** | 217.0 |
| **Refund Request** | 0.9829 | 1.0000 | **0.9914** | 115.0 |
| **Technical Issue** | 1.0000 | 0.9968 | **0.9984** | 945.0 |

---

## 5. Error Analysis & Confusion Matrix Findings

* **Misclassified Test Samples**: `3` out of `1,694` test cases (0.18% error rate).
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
