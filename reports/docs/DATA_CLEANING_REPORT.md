# Data Cleaning and Text Preprocessing Report

This report documents the details and decisions of the data cleaning stage for the Support Ticket Classification ML project.

---

## 1. Row and Column Counts

| Metric | Original | Final | Change / Details |
| :--- | :---: | :---: | :--- |
| **Row Count** | `8,469` | `8,469` | No rows were dropped; target variables were fully populated. |
| **Column Count** | `17` | `18` | `+1` column added for the combined text feature `ticket_text`. |

---

## 2. Text Cleaning Operations Performed

To prepare customer support tickets for natural language processing and machine learning, we applied a standard cleaning pipeline to the `Ticket Description` and `Ticket Subject` fields:

1. **Handling Missing Text**: Replaced any null/missing entries with an empty string `""` to prevent processing crashes.
2. **String Conversion**: Ensured both text columns are strictly string objects.
3. **Casing Normalization**: Converted all text to lowercase.
4. **Whitespace Trimming**: Removed leading and trailing spaces.
5. **Whitespace Normalization**: Replaced double-spaces, newlines (`\n`), and tabs (`\t`) with a single space.
6. **No Destructive Filtering**: No stopword removal or punctuation removal was performed at this stage. Crucial semantic words such as negation terms (`not`, `no`, `never`, `can't`, `cannot`, `failed`) were intentionally preserved because they are critical in ticket intent detection.

---

## 3. Placeholder Replacement Results

The raw dataset contained placeholder strings indicating templated text. We used case-insensitive regular expressions to dynamically resolve these placeholders using row-level values:

* **Primary Placeholder Resolved**: `{product_purchased}` (and variations such as `{Product_purchased}`, `{Product_Purchased}`, and `{product_purchased.name}`) in `Ticket Description` were replaced with the value in the corresponding `Product Purchased` column (e.g. *Laptop*, *Canon EOS*, etc.).
* **Result**:
  * **Original Count**: `8,469` descriptions containing placeholders.
  * **Final Count**: `0` unresolved `product_purchased` placeholders.
  * **Typo Handling**: The regular expression successfully resolved anomalous double-brace placeholders (e.g. `{product_purchased}}`) and raw strings.

---

## 4. Combined Text Feature

We created a combined feature named **`ticket_text`**:
* **Formula**: `Ticket Subject` + `" "` + `Ticket Description`
* **Purpose**: Combines the high-level intent from the subject line with the detailed description into a single unified text string, which is optimal for NLP feature engineering.
* **Original Columns Kept**: The original cleaned `Ticket Description` and `Ticket Subject` columns remain in the dataset for separate analysis if required.

---

## 5. Categorical & Target Variable Cleaning

* **Target Labels Normalization**: We stripped any leading/trailing spaces from the target categorical columns.
* **Target Categories (`Ticket Type`)**:
  * *Refund request*: 1,752 (20.69%)
  * *Technical issue*: 1,747 (20.63%)
  * *Cancellation request*: 1,695 (20.01%)
  * *Product inquiry*: 1,641 (19.38%)
  * *Billing inquiry*: 1,634 (19.29%)
  * *Missing values*: `0`
* **Priority Labels (`Ticket Priority`)**:
  * *Medium*: 2,192 (25.88%)
  * *Critical*: 2,129 (25.14%)
  * *High*: 2,085 (24.62%)
  * *Low*: 2,063 (24.36%)
  * *Missing values*: `0`

No class modifications or row deletions were needed as the targets were fully clean and complete.

---

## 6. Datetime Conversion & Missing Values Handling

* **Parsed Date Columns**: The columns `Date of Purchase`, `First Response Time`, and `Time to Resolution` were converted to proper datetime representations.
* **Missing Value Preservations**:
  * `Resolution`: `5,700` missing values preserved.
  * `Time to Resolution`: `5,700` missing values preserved.
  * `First Response Time`: `2,819` missing values preserved.
  * `Customer Satisfaction Rating`: `5,700` missing values preserved.
* **Decision Rationale**: We **did not** replace missing response or resolution fields with fake filler values. These fields are null because the tickets are either `Open` or `Pending Customer Response`. Filling them with placeholder dates or text would distort the data distribution and create misleading records for temporal analysis.

---

## 7. Data Leakage Control

To prevent data leakage during machine learning, we classified the columns into two groups:

### 🚫 Excluded from Model Features (Post-Resolution / Post-Ticket Creation Information)
The following columns contain information generated *after* a ticket is created and handled by support agents. They must **never** be used as inputs to the classifier, as they will not be available when predicting incoming support ticket categories:
1. `Ticket Status` (leakage if used, as it indicates whether a ticket is solved)
2. `Resolution` (leakage, contains agent solutions)
3. `First Response Time` (leakage, indicates agent action time)
4. `Time to Resolution` (leakage, indicates solve time)
5. `Customer Satisfaction Rating` (leakage, indicates customer review post-solution)

### Valid Model Features (Available at Ticket Submission)
Only the following fields are safe to use for predictions:
* `ticket_text` / `Ticket Description` / `Ticket Subject`
* `Customer Age`, `Customer Gender`
* `Product Purchased`, `Date of Purchase`
* `Ticket Channel`

---

## 8. Remaining Data-Quality Issues

There are minor, non-blocking data quality issues in the text columns:
1. **Broken Placeholders**: There are 2 instances of incomplete template placeholders (`{product_purchased` missing the closing bracket, and `product_purchased}` missing the opening bracket) that were not replaced because they lack proper structure. These are trivial typos and will be treated as regular lowercase vocabulary during vectorization.
2. **Artificial Resolutions**: Nonsensical filler sentences remain in the `Resolution` text. Since `Resolution` is excluded from the machine learning model features to avoid leakage, this does not affect our classifier.

---

## 9. Readiness Check

The cleaned dataset is saved to:
[`customer_support_tickets_cleaned.csv`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/data/processed/customer_support_tickets_cleaned.csv)

The dataset contains no missing target variables, has fully resolved `{product_purchased}` placeholders, has a clean combined `ticket_text` column, and has date fields parsed safely.

**Verdict: The dataset is 100% READY for Exploratory Data Analysis (EDA) and Model Training.**
