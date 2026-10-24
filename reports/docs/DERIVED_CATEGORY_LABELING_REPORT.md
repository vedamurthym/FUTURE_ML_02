# Derived Category Labeling Report (Stage 5B)

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

* **Total Records Labeled**: `8,469`
* **Unclear Tickets Count**: `0` (`0.00%`)

### Class Counts & Percentages:

| Derived Category | Count | Percentage (%) |
| :--- | :---: | :---: |
| **Technical Issue** | 4,726.0 | 55.80%
| **Product Inquiry** | 1,084.0 | 12.80%
| **Refund Request** | 576.0 | 6.80%
| **Delivery & Order Inquiry** | 561.0 | 6.62%
| **Billing & Payment** | 526.0 | 6.21%
| **Account & Access** | 509.0 | 6.01%
| **Cancellation Request** | 487.0 | 5.75%

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
