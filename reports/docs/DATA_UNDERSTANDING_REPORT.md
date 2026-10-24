# Data Understanding Report

This report documents the initial data exploration and profiling results for the Support Ticket Classification ML project dataset.

---

## 1. Dataset Overview

* **File Name**: `customer_support_tickets.csv`
* **Total Rows**: `8,469`
* **Total Columns**: `17`

### Column Inventory & Data Types

| Column Name | Data Type (Parsed) | Description / Role |
| :--- | :--- | :--- |
| `Ticket ID` | `int64` | Unique identifier for each ticket |
| `Customer Name` | `object` (string) | Name of the customer |
| `Customer Email` | `object` (string) | Email of the customer |
| `Customer Age` | `int64` | Age of the customer |
| `Customer Gender` | `object` (string) | Gender of the customer |
| `Product Purchased` | `object` (string) | Product relevant to the ticket |
| `Date of Purchase` | `object` (string) | Date the product was bought (needs parse to datetime) |
| `Ticket Type` | `object` (string) | **Category/Class Column** (Label) |
| `Ticket Subject` | `object` (string) | Ticket subject line (secondary text field) |
| `Ticket Description` | `object` (string) | **Ticket Text Column** (Primary source text) |
| `Ticket Status` | `object` (string) | Status of the ticket (Open, Pending Customer Response, Closed) |
| `Resolution` | `object` (string) | Solution details provided for closed tickets |
| `Ticket Priority` | `object` (string) | **Priority Column** (Low, Medium, High, Critical) |
| `Ticket Channel` | `object` (string) | Channel through which ticket was submitted |
| `First Response Time` | `object` (string) | Date/time of first agent reply (needs parse to datetime) |
| `Time to Resolution` | `object` (string) | Date/time of final resolution (needs parse to datetime) |
| `Customer Satisfaction Rating` | `float64` | CSAT rating (1 to 5) |

---

## 2. Missing Values & Duplicate Records

* **Duplicate Rows**: `0` duplicate rows detected.
* **Missing Value Analysis**:

| Column Name | Missing Count | Percentage (%) |
| :--- | :---: | :---: |
| `Resolution` | 5,700 | 67.30% |
| `Time to Resolution` | 5,700 | 67.30% |
| `Customer Satisfaction Rating` | 5,700 | 67.30% |
| `First Response Time` | 2,819 | 33.29% |
| *(All other 13 columns)* | 0 | 0.00% |

### Key Correlation Findings:
1. **Resolution** and **Time to Resolution** are missing for exactly `5,700` tickets. This maps perfectly to tickets where `Ticket Status` is either `Open` (`2,819`) or `Pending Customer Response` (`2,881`). They are only present for `Closed` tickets (`2,769`).
2. **First Response Time** is missing for exactly `2,819` tickets, matching the count of `Open` tickets. This indicates that open tickets have not yet received a response.

---

## 3. Key Target Columns Identified

1. **Ticket Text Column**: `Ticket Description`
   * Contains the descriptive text submitted by the customer.
   * A secondary text field is `Ticket Subject`, which could be concatenated or analyzed in future stages.
2. **Category / Class Column**: `Ticket Type`
   * Target for the classification task.
   * Has 5 classes, which are evenly distributed (approx. 19% - 21% each).
3. **Priority Column**: `Ticket Priority`
   * Secondary label or metadata for classifying urgency.
   * Has 4 priority classes (Low, Medium, High, Critical) which are also evenly distributed (approx. 24% - 26% each).

---

## 4. Ticket Description Length Analysis

Character and word statistics computed on the primary text column `Ticket Description`:

| Metric | Length in Characters | Length in Words |
| :--- | :---: | :---: |
| **Minimum** | 151 | 21 |
| **Maximum** | 397 | 63 |
| **Average (Mean)** | 289.82 | 46.47 |
| **Median** | 298.0 | 49.0 |

* **Insight**: The text descriptions are relatively short and consistent in length. There are no extreme outlier texts (e.g. thousands of words long), which makes text processing more predictable.

---

## 5. Main Data Quality Issues Identified

During initial profiling, the following core data-quality issues were found:

1. **Template Placeholders in Descriptions**:
   The `Ticket Description` column contains raw curly-brace template placeholders, specifically `{product_purchased}` (e.g., *"I'm having an issue with the {product_purchased}. Please assist."*). These placeholders should be resolved dynamically using the values from the `Product Purchased` column.
2. **Artificial/Nonsensical Resolution Text**:
   The `Resolution` column contains placeholder-style sentences or low-quality artificial text (e.g., *"Case maybe show recently my computer follow."*). This might limit the use of the resolution text as a meaningful target for text generation, though ticket type classification remains unaffected.
3. **Unparsed Date Columns**:
   The columns `Date of Purchase`, `First Response Time`, and `Time to Resolution` are loaded as string types. They need parsing to proper datetime formats to enable date/time analysis or feature engineering.
4. **Conditional Nulls**:
   The columns `Resolution`, `Time to Resolution`, `First Response Time`, and `Customer Satisfaction Rating` have structural missing values linked to `Ticket Status`. These are not random errors but systematic database characteristics.

---

## 6. Categorical Column Distributions

### Ticket Type (Target Variable)
* **Refund request**: 1,752 (20.69%)
* **Technical issue**: 1,747 (20.63%)
* **Cancellation request**: 1,695 (20.01%)
* **Product inquiry**: 1,641 (19.38%)
* **Billing inquiry**: 1,634 (19.29%)
* *Status*: Balanced distribution. No class imbalance issues.

### Ticket Priority
* **Medium**: 2,192 (25.88%)
* **Critical**: 2,129 (25.14%)
* **High**: 2,085 (24.62%)
* **Low**: 2,063 (24.36%)
* *Status*: Highly balanced.

### Ticket Status
* **Pending Customer Response**: 2,881 (34.02%)
* **Open**: 2,819 (33.29%)
* **Closed**: 2,769 (32.70%)
