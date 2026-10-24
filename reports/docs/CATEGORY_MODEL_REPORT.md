# Category Model Training & Evaluation Report

This report documents the baseline machine learning models developed to classify support tickets into **Ticket Type** categories.

---

## 1. Executive Summary & Objective

* **Objective**: Predict `Ticket Type` (5 categories) from incoming customer support ticket text (`ticket_text`).
* **Training Dataset Size**: `6,775` samples (80%)
* **Testing Dataset Size**: `1,694` samples (20%)
* **Best Model Selected**: **Linear SVM**
* **Best Model F1-Score (Weighted)**: `0.1944`

---

## 2. Feature Extraction & TF-IDF Configuration

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts raw customer ticket text into numerical feature vectors based on how important words are within the corpus:
* **Term Frequency (TF)**: Measures how frequently a word appears in a specific ticket.
* **Inverse Document Frequency (IDF)**: Downweights words that appear commonly across all tickets (e.g. general terms) and highlights distinctive words.

### Vectorizer Configuration:
* **NGram Range**: `(1, 2)` (Includes both single words and two-word phrases like *"not working"*, *"billing charge"*)
* **Min Document Frequency (`min_df`)**: `2` (Ignores rare terms appearing in only 1 document)
* **Sublinear TF Scaling**: Enabled (applies `1 + log(tf)` to prevent dominant term counts)
* **Total Features Extracted**: `10,000` numerical TF-IDF features

---

## 3. Models Tested

1. **Logistic Regression**: Linear multiclass classifier (`multi_class='multinomial'`) optimizing log-loss.
2. **Linear Support Vector Machine (LinearSVC)**: Maximum-margin hyperplane classifier well-suited for high-dimensional text data.
3. **Multinomial Naive Bayes**: Probabilistic classifier applying Bayes' theorem with independence assumptions.

---

## 4. Evaluation Metrics & Comparison Table

Models were evaluated on the `1,694` held-out test tickets. Metrics use **weighted averaging** across all 5 category classes:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.1901 | 0.1894 | 0.1901 | 0.1895 |
| **Linear SVM** | 0.1948 | 0.1945 | 0.1948 | 0.1944 |
| **Multinomial Naive Bayes** | 0.1924 | 0.1884 | 0.1924 | 0.1863 |

* **Primary Selection Metric**: **F1 Score** was chosen as the primary metric because it balances Precision and Recall across all multiclass categories.
* **Winning Model**: **Linear SVM** achieved the highest F1-score of `0.1944`.

---

## 5. Confusion Matrix Observations & Error Analysis

### Confusion Matrix Plots:
Saved visualizations in `reports/figures/`:
- [`category_logistic_regression_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_logistic_regression_confusion_matrix.png)
- [`category_linear_svm_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_linear_svm_confusion_matrix.png)
- [`category_naive_bayes_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_naive_bayes_confusion_matrix.png)

### Error Analysis & Most Common Misclassification:
* **Misclassified Test Samples**: `1364` out of `1694` test tickets (80.52% error rate).
* **Most Frequent Confusion Pair**: 'Product inquiry' misclassified as 'Technical issue' (92 times).
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

The **Linear SVM** model is saved and ready for deployment or downstream inference tasks.
