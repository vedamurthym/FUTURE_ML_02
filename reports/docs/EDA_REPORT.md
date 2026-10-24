# Exploratory Data Analysis (EDA) Report

This report presents a thorough exploratory analysis of the cleaned customer support ticket dataset (`customer_support_tickets_cleaned.csv`).

---

## 1. Dataset Overview

* **Total Records (Rows)**: `8,469`
* **Total Features (Columns)**: `18`
* **Duplicate Rows**: `0`
* **Primary Text Column**: `ticket_text` (Combined `Ticket Subject` and `Ticket Description`)
* **Primary Target Column**: `Ticket Type` (5 categories)
* **Secondary Target Column**: `Ticket Priority` (4 priority levels)
* **Missing Values**:
  * `Resolution`: 5,700 missing (67.30%)
  * `Time to Resolution`: 5,700 missing (67.30%)
  * `Customer Satisfaction Rating`: 5,700 missing (67.30%)
  * `First Response Time`: 2,819 missing (33.29%)
  * All other 14 columns (including text and target labels) have `0` missing values.

---

## 2. Ticket Category Analysis (`Ticket Type`)

The dataset contains 5 unique support ticket categories:

| Ticket Category | Count | Percentage (%) | Description / Intent |
| :--- | :---: | :---: | :--- |
| **Refund request** | 1,752 | 20.69% | Customers requesting money back or order returns. |
| **Technical issue** | 1,747 | 20.63% | Customers reporting software/hardware glitches or crashes. |
| **Cancellation request** | 1,695 | 20.01% | Customers wanting to cancel subscriptions or orders. |
| **Product inquiry** | 1,641 | 19.38% | General questions regarding product features or setup. |
| **Billing inquiry** | 1,634 | 19.29% | Charges, invoice clarifications, or payment issues. |

* **Most Common Category**: `Refund request` (1,752 tickets, 20.69%)
* **Least Common Category**: `Billing inquiry` (1,634 tickets, 19.29%)
* **Visualization**: Saved bar chart at [`ticket_category_distribution.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/ticket_category_distribution.png).

---

## 3. Priority Analysis (`Ticket Priority`)

Tickets are classified into 4 priority levels:

| Priority Level | Count | Percentage (%) |
| :--- | :---: | :---: |
| **Low** | 2,063 | 24.36% |
| **Medium** | 2,192 | 25.88% |
| **High** | 2,085 | 24.62% |
| **Critical** | 2,129 | 25.14% |

* **Visualization**: Saved bar chart at [`ticket_priority_distribution.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/ticket_priority_distribution.png).

---

## 4. Category vs. Priority Relationship

Cross-tabulation count matrix between `Ticket Type` and `Ticket Priority`:

| Ticket Category | Low | Medium | High | Critical | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Billing inquiry** | 398 | 434 | 382 | 420 | 1,634 |
| **Cancellation request** | 414 | 460 | 398 | 423 | 1,695 |
| **Product inquiry** | 398 | 441 | 399 | 403 | 1,641 |
| **Refund request** | 440 | 420 | 448 | 444 | 1,752 |
| **Technical issue** | 413 | 437 | 458 | 439 | 1,747 |

### Row-Wise Distribution (% within Category):

| Ticket Category | Low (%) | Medium (%) | High (%) | Critical (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Billing inquiry** | 24.36% | 26.56% | 23.38% | 25.70% |
| **Cancellation request** | 24.42% | 27.14% | 23.48% | 24.96% |
| **Product inquiry** | 24.25% | 26.87% | 24.31% | 24.56% |
| **Refund request** | 25.11% | 23.97% | 25.57% | 25.34% |
| **Technical issue** | 23.64% | 25.01% | 26.22% | 25.13% |

### Key Observations:
1. **Uniform Priority Distribution**: Priority levels are almost perfectly uniformly distributed across all categories (approx. 23% - 27% per cell).
2. **No Dominant Category-Priority Link**: No single category is disproportionately marked as `Critical` or `Low`. For instance, `Technical issue` has 439 Critical tickets (25.13%), which is virtually identical to `Billing inquiry` (25.70%).
3. **Visualization**: Saved heatmap at [`category_vs_priority.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/category_vs_priority.png).

---

## 5. Text Length Analysis (`ticket_text`)

Character length metrics computed on the primary model input `ticket_text`:

* **Minimum Length**: `148` characters
* **Maximum Length**: `419` characters
* **Mean Length**: `297.25` characters
* **Median Length**: `306.00` characters
* **Standard Deviation**: `44.32` characters

### Outlier Detection (IQR Method):
* **Lower Threshold (< 202.0 chars)**: `455` outlier tickets (unusually short)
* **Upper Threshold (> 402.0 chars)**: `3` outlier tickets (unusually long)
* **Visualization**: Saved histogram at [`ticket_text_length_distribution.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/ticket_text_length_distribution.png).

---

## 6. Common Word Analysis

Top 20 most frequent terms across all ticket texts after stopword and pronoun contraction removal (with negations strictly retained):

| Rank | Word | Occurrences | Role / Meaning |
| :---: | :--- | :---: | :--- |
| 1 | `product` | 3,899 | Core domain noun |
| 2 | `not` | 2,280 | **Negation term** (critical non-working indicator) |
| 3 | `data` | 2,155 | Data / storage reference |
| 4 | `software` | 2,129 | Application / OS reference |
| 5 | `account` | 1,971 | User account context |
| 6 | `steps` | 1,390 | Troubleshooting steps |
| 7 | `noticed` | 1,208 | Issue observation term |
| 8 | `persists` | 1,178 | Ongoing issue status |
| 9 | `hardware` | 1,176 | Device physical component |
| 10 | `resolve` | 1,165 | Issue resolution keyword |
| 11 | `request` | 1,157 | Support request term |
| 12 | `update` | 1,156 | Firmware/software update |
| 13 | `would` | 1,121 | Customer request phrasing |
| 14 | `device` | 1,106 | Equipment reference |
| 15 | `unable` | 1,088 | Capability failure term |
| 16 | `compatibility` | 1,064 | Interoperability issue |
| 17 | `lg` | 1,058 | Brand identifier |
| 18 | `could` | 1,057 | Customer inquiry phrasing |
| 19 | `sony` | 1,056 | Brand identifier |
| 20 | `microsoft` | 1,019 | Brand identifier |

* **Visualization**: Saved horizontal bar chart at [`top_20_words.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/top_20_words.png).

---

## 7. Category-Specific Observations

Top representative keywords identified per ticket category:

1. **Technical issue**: `product`, `not`, `software`, `data`, `account`, `steps`, `resolve`, `persists`, `noticed`, `request`.
2. **Billing inquiry**: `product`, `not`, `software`, `data`, `account`, `noticed`, `steps`, `would`, `resolve`, `update`.
3. **Cancellation request**: `product`, `not`, `data`, `software`, `account`, `steps`, `hardware`, `request`, `resolve`, `noticed`.
4. **Product inquiry**: `product`, `data`, `software`, `not`, `account`, `steps`, `persists`, `request`, `unable`, `hardware`.
5. **Refund request**: `product`, `not`, `data`, `software`, `account`, `steps`, `noticed`, `hardware`, `update`, `sony`.

* **Insight**: Ticket text across categories relies heavily on shared synthetic ticket templates in this dataset. Vectorization methods with n-grams will be key to capturing subtle word combinations.

---

## 8. Priority-Specific Observations

Text characteristics compared across priority levels:

| Priority Level | Average Character Length | Average Word Count | Key Frequent Terms |
| :--- | :---: | :---: | :--- |
| **Low** | 295.6 chars | 50.2 words | `product`, `not`, `software`, `data`, `account` |
| **Medium** | 297.4 chars | 50.6 words | `product`, `not`, `software`, `data`, `account` |
| **High** | 299.4 chars | 50.9 words | `product`, `data`, `not`, `software`, `account` |
| **Critical** | 296.5 chars | 50.5 words | `product`, `not`, `data`, `software`, `account` |

* **Observation**: Text length and word choices are virtually identical across all priority levels (mean length ~295–299 characters). Urgency is not encoded in ticket length.

---

## 9. Class Imbalance Analysis

Imbalance metrics calculated for target variables:

### 1. `Ticket Type` (5 classes)
* **Majority Class**: `Refund request` (1,752 tickets, 20.69%)
* **Minority Class**: `Billing inquiry` (1,634 tickets, 19.29%)
* **Imbalance Ratio**: `1.072` (Majority / Minority)

### 2. `Ticket Priority` (4 classes)
* **Majority Class**: `Medium` (2,192 tickets, 25.88%)
* **Minority Class**: `Low` (2,063 tickets, 24.36%)
* **Imbalance Ratio**: `1.063` (Majority / Minority)

* **Assessment**: Both targets are exceptionally well-balanced. There is **no class imbalance problem**. Standard accuracy and macro-F1 metrics will be reliable.

---

## 10. Data Leakage Considerations

To enforce strict ML boundaries, post-resolution columns were audited:
* `Resolution` (Present, 67.30% null) -> **EXCLUDED**
* `Time to Resolution` (Present, 67.30% null) -> **EXCLUDED**
* `Customer Satisfaction Rating` (Present, 67.30% null) -> **EXCLUDED**
* `First Response Time` (Present, 33.29% null) -> **EXCLUDED**
* `Ticket Status` (Present) -> **EXCLUDED**

* **Status**: None of these fields will be passed to model feature pipelines. Models will strictly take `ticket_text` (and pre-submission metadata if needed) as input.

---

## 11. Summary of Important Findings

1. **Clean & Balanced Target**: The dataset is completely clean with 0 missing labels and uniform class distributions (~20% per category, ~25% per priority).
2. **Negation Terms**: `not` is among the top 2 most frequent terms (2,280 occurrences). Preserving negations during text preprocessing is essential.
3. **Consistent Text Length**: Text length is highly consistent (mean = 297 characters, std = 44.3 chars) with no extreme multi-thousand-character outliers.

---

## 12. Recommendations for ML Preparation (Stage 4)

1. **Feature Extraction**: Use TF-IDF vectorization with `ngram_range=(1, 2)` or `(1, 3)` to capture multi-word phrases.
2. **Preprocessing**: Retain contractions and negations in stopword lists.
3. **Cross-Validation**: Use 5-Fold Stratified K-Fold cross-validation.
4. **Evaluation Metrics**: Macro F1-Score, Micro F1-Score, and Multi-class Confusion Matrices.
