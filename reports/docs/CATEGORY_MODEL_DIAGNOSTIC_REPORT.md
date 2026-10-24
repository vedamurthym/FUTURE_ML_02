# Category Model Diagnostic Report (Stage 5A)

This report details the diagnostic investigation conducted to determine why the Ticket Category Classification model achieved random-chance performance (~19.48% accuracy / weighted F1 score across 5 classes).

---

## 1. Core Diagnostic Questions & Answers

### Q1: Why might the model have such low performance (~19.48%)?
* **Answer**: In the synthetic Kaggle dataset (`customer_support_tickets.csv`), the `Ticket Type` column values (*Refund request*, *Technical issue*, *Cancellation request*, *Product inquiry*, *Billing inquiry*) were generated as **independent random noise** during dataset creation.
* **Empirical Proof**:
  * Random chance for 5 equal classes is `1 / 5 = 20.00%`.
  * Logistic Regression achieved `18.95%` F1, Linear SVM achieved `19.44%` F1, and Naive Bayes achieved `18.63%` F1.
  * Every `Ticket Subject` line (e.g. *"Refund request"*, *"Payment issue"*, *"Data loss"*) is split almost perfectly 20% / 20% / 20% / 20% / 20% across all 5 `Ticket Type` categories.
  * When attempting to predict `Ticket Subject` (16 classes) from `ticket_text`, the exact same Linear SVM model achieves **`100.0%` Accuracy (1.0000 F1)**! This proves the text pipeline and model architecture work perfectly, but the `Ticket Type` column has 0 mathematical relationship with ticket text or subjects.

### Q2: Are the ticket descriptions correctly matched to Ticket Type?
* **Answer**: **No.** In the raw CSV data, ticket descriptions describing account credentials or network crashes are frequently assigned to *Billing inquiry* or *Refund request* labels, and vice versa. The label assignment in `Ticket Type` does not correlate with text semantics.

### Q3: Are the categories distinguishable from the text?
* **Answer**: The 5 `Ticket Type` classes are **not distinguishable** because their text distributions and keyword frequency rankings are identical across all 5 classes (each class shares the same top vocabulary: *product*, *not*, *data*, *software*, *account*, *steps*). Conversely, the 16 `Ticket Subject` categories are 100% distinguishable from the text.

### Q4: Is the preprocessing damaging the text?
* **Answer**: **No.** Text preprocessing preserves negations (*not*, *no*, *never*, *cannot*, *failed*) and normalizes whitespace without stripping critical intent. Preprocessed text predicts `Ticket Subject` with `100.0%` accuracy, confirming the preprocessing pipeline is completely clean.

### Q5: Is there a train/test data problem?
* **Answer**: **No.** Train/test splits were created with 80/20 stratification (`random_state=42`). Class ratios are identical between train and test sets (`category_train.csv` and `category_test.csv`).

### Q6: Does including Ticket Subject improve performance when predicting Ticket Type?
* **Answer**: **No.**
  * `ticket_text` alone predicting `Ticket Type`: Acc = `0.1948`, F1 = `0.1946`
  * Formatted `Subject + Description` predicting `Ticket Type`: Acc = `0.2019`, F1 = `0.2016`
  * `Ticket Subject` alone predicting `Ticket Type`: Acc = `0.1783`, F1 = `0.1743`
  * Adding `Ticket Subject` does not help predict `Ticket Type` because `Ticket Subject` is also independent of `Ticket Type`.

### Q7: What should we change before final model training?
* **Answer**:
  1. For `Ticket Type` classification on this specific synthetic Kaggle dataset, an accuracy of ~19.5% represents the true mathematical ceiling because `Ticket Type` is uncorrelated noise.
  2. The pipeline, vectorizer, and models implemented in Stage 5 are operating at optimal capability given the target.
  3. We should proceed to Stage 6 (Priority Classification Model) to evaluate whether `Ticket Priority` contains learnable signals.

---

## 2. Model Performance Summary

| Feature Representation | Target Variable | Model | Accuracy | Weighted F1 |
| :--- | :--- | :--- | :---: | :---: |
| `ticket_text` | `Ticket Type` (5 classes) | Linear SVM | `0.1948` | `0.1946` |
| Formatted `Subject+Desc` | `Ticket Type` (5 classes) | Linear SVM | `0.2019` | `0.2016` |
| `Ticket Subject` | `Ticket Type` (5 classes) | Linear SVM | `0.1783` | `0.1743` |
| `ticket_text` | `Ticket Subject` (16 classes) | Linear SVM | `1.0000` | `1.0000` |

---

## 3. Conclusion & Next Steps

* **Diagnosis Verdict**: The machine learning pipeline, text preprocessing, feature engineering, and model training in Stage 5 are fully correct. The ~19.5% accuracy ceiling is due to random label generation in the `Ticket Type` column of this synthetic dataset.
* **Action Plan**: Proceed to **Stage 6 (Ticket Priority Classification Model)**.
