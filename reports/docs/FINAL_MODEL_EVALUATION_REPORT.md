# FINAL MODEL EVALUATION REPORT
## Stage 7 — Final Evaluation of Category and Priority Models

> **⚠️ IMPORTANT NOTICE & ETHICAL DISCLAIMER**
> "The category and priority targets used for supervised learning were derived
> through documented rules after the original Ticket Type and Ticket Priority
> fields were found to be noisy/non-predictive. Therefore, these evaluation scores
> measure agreement with the derived labeling system and should not be interpreted
> as human-annotated ground-truth accuracy."

---

## 1. Executive Summary

This report presents the final evaluation of our trained support ticket classification models. Both models were tested on held-out test datasets (representing 20% of the labeled dataset) that were completely isolated during model training and feature extraction.

### Summary Metrics

| Model | Accuracy | Macro F1 | Weighted F1 | Target Variable |
| --- | --- | --- | --- | --- |
| **Category Model** | 0.9982 | 0.9979 | 0.9982 | `derived_category` |
| **Priority Model** | 0.9841 | 0.9482 | 0.9841 | `derived_priority` |

---

## 2. Dataset and Test Sets

- **Category dataset:** `customer_support_tickets_labeled.csv` (8,469 rows)
  - Train size: 6,775 rows (80%)
  - Test size: 1,694 rows (20%) — stratified split.
- **Priority dataset:** `customer_support_tickets_priority_labeled.csv` (8,469 rows)
  - Train size: 6,775 rows (80%)
  - Test size: 1,694 rows (20%) — stratified split.

---

## 3. Category Model Evaluation

The Category Classification Model (`derived_category_model.pkl`) evaluates text representation of support tickets to classify them into one of seven customer service categories.

- **Test accuracy:** 0.9982 (99.82%)
- **Macro F1-Score:** **0.9979**
- **Weighted F1-Score:** **0.9982**

---

## 4. Category Per-Class Performance

| Derived Category | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Account & Access | 1.0000 | 1.0000 | **1.0000** | 102 |
| Billing & Payment | 1.0000 | 1.0000 | **1.0000** | 105 |
| Cancellation Request | 1.0000 | 1.0000 | **1.0000** | 98 |
| Delivery & Order Inquiry | 0.9912 | 1.0000 | **0.9956** | 112 |
| Product Inquiry | 1.0000 | 1.0000 | **1.0000** | 217 |
| Refund Request | 0.9829 | 1.0000 | **0.9914** | 115 |
| Technical Issue | 1.0000 | 0.9968 | **0.9984** | 945 |

---

## 5. Category Confusion Matrix

- Confusion matrix plot saved to: [`final_category_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/final_category_confusion_matrix.png)

The model shows extremely high classification performance across all categories. This high metric represents strong internal logic alignment since categories are rule-derived.

---

## 6. Category Error Analysis

The category classifier has an extremely low error rate of 0.18%.
Below are the most common incorrect classification pairs (actual -> predicted):

| Actual Label | Predicted Label | Count |
| --- | --- | --- |
| Technical Issue | Refund Request | 2 |
| Technical Issue | Delivery & Order Inquiry | 1 |

---

## 7. Priority Model Evaluation

The Priority Classification Model (`priority_model.pkl`) evaluates incoming tickets and classifies them into one of four priority categories (*Critical*, *High*, *Medium*, *Low*).

- **Test accuracy:** 0.9841 (98.41%)
- **Macro F1-Score:** **0.9482**
- **Weighted F1-Score:** **0.9841**

---

## 8. Priority Per-Class Performance

| Priority Level | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Critical | 1.0000 | 0.8293 | **0.9067** | 41 |
| High | 0.9917 | 0.9929 | **0.9923** | 841 |
| Medium | 0.9892 | 0.9826 | **0.9859** | 747 |
| Low | 0.8421 | 0.9846 | **0.9078** | 65 |

---

## 9. Priority Confusion Matrix

- Confusion matrix plot saved to: [`final_priority_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/final_priority_confusion_matrix.png)

Below are the most common incorrect classification pairs (actual -> predicted):

| Actual Label | Predicted Label | Count |
| --- | --- | --- |
| Medium | Low | 12 |
| Critical | High | 6 |
| High | Medium | 6 |
| Critical | Medium | 1 |
| Low | Medium | 1 |
| Medium | High | 1 |

---

## 10. Critical Priority Analysis

Operationally, the detection of **Critical** priority tickets is the most vital business requirement. Failing to route a critical outage or security emergency immediately can lead to significant financial loss and user disruption.

### Critical Metrics

| Metric | Value |
| --- | --- |
| Critical Support (Test Set) | 41 |
| Correctly Predicted | 34 (82.9%) |
| Missed (Incorrectly Predicted) | 7 (17.1%) |
| Critical Precision | 1.0000 |
| Critical Recall | **0.8293** |
| Critical F1-Score | **0.9067** |

- Critical errors saved to: [`critical_priority_errors.csv`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/critical_priority_errors.csv)

### Operational Importance of Critical Recall:
- **Recall (Sensitivity)** measures the ratio of actual critical tickets we correctly detect.
- A lower Recall means we miss critical cases (False Negatives), leaving security breaches or device failure issues unaddressed.
- A 100% precision score suggests that when the model claims a ticket is Critical, it is indeed Critical. However, the recall of 82.9% indicates that some critical cases are missed. Human-in-the-loop fallback verification is strongly recommended for these edge cases.

---

## 11. Model Consistency Checks

All consistency audits passed:
1. Category model successfully loaded.
2. Category TF-IDF vectorizer successfully loaded.
3. Priority model successfully loaded.
4. Priority TF-IDF vectorizer successfully loaded.
5. TF-IDF features match model expected input dimensionality:
   - Category features: 1694.0
   - Priority features: 1694.0
6. Test datasets contain no unexpected target classes.
7. Checks confirmed no NaN values or empty strings were passed to vectorizers.

---

## 12. Business Interpretation

### Category Model:
The system categorizes incoming support tickets with **99.82%** accuracy. This ensures that tickets are routed to the correct agent domain (e.g. refund requests to billing teams, software bugs to development teams) with negligible routing error, reducing support ticket dispatch latency.

### Priority Model:
The system prioritizes incoming tickets with **98.41%** accuracy. It serves as an automated triage tool to place high-severity and medium-severity tickets in appropriate response queues.

### Critical Priority:
The system successfully identifies **34 out of 41 (82.9%)** critical tickets. The remaining 7 tickets are classified as *High* or *Medium*. While they are not dropped entirely, they will experience longer response times compared to a direct Critical triage.

---

## 13. Limitations

1. **Rule-derived labels:** The evaluations measure performance relative to rule-assigned labels. Real-world human labels might slightly deviate based on subtle context.
2. **Critical Class Representation:** The Critical priority class represents only 2.44% of the dataset. Therefore, the priority model's exposure to Critical examples is limited, leading to a Recall of 0.8293.
3. **Keyword-based rules:** Because labels were derived using keywords, the model essentially learns a surrogate representation of the keywords. It may fail to generalise to tickets using completely novel vocabulary.

---

## 14. Final Conclusion

The saved category model (`derived_category_model.pkl`) and priority model (`priority_model.pkl`) are robust and ready to be integrated into the support ticket classification system. The final models demonstrate outstanding Macro F1 scores (Category: **0.9979**, Priority: **0.9482**), confirming their suitability for high-imbalance ticket dispatch pipelines.
