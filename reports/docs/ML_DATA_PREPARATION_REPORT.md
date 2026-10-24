# ML Data Preparation Report

This report documents the machine learning data preparation phase for the Support Ticket Classification project.

---

## 1. Dataset Row Counts & Removal Summary

* **Starting Row Count**: `8,469`
* **Final Prepared Rows**: `8,469`
* **Total Rows Removed**: `0`
  * Missing/Empty Category Target: `0`
  * Missing/Empty Priority Target: `0`
  * Missing/Empty Ticket Text: `0`

---

## 2. Feature & Target Definitions

### ML Task A: Category Classification
* **Feature Input (`X`)**: `ticket_text` (Combined Subject + Description text)
* **Target Label (`y_category`)**: `Ticket Type` (5 classes)

### ML Task B: Priority Classification
* **Feature Input (`X`)**: `ticket_text` (Combined Subject + Description text)
* **Target Label (`y_priority`)**: `Ticket Priority` (4 classes)

---

## 3. Data Leakage Control & Excluded Columns

The following post-resolution columns were **strictly excluded** from ML prediction features to prevent data leakage:
* `Resolution`
* `Time to Resolution`
* `Customer Satisfaction Rating`
* `First Response Time`
* `Ticket Status`

---

## 4. Train / Test Split Parameters

* **Test Set Percentage**: `20.0%` (`test_size = 0.20`)
* **Training Set Percentage**: `80.0%`
* **Random State Seed**: `42` (ensures exact reproducibility across runs)
* **Stratification Method**: Stratified random sampling based on target class labels (`stratify = y`)

---

## 5. Target A: Ticket Category Class Distribution

* **Category Training Rows**: `6,775`
* **Category Testing Rows**: `1,694`

### Train vs. Test Class Comparison:

| Category Class | Train Count | Train (%) | Test Count | Test (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Refund request** | 1,401.0 | 20.68% | 351.0 | 20.72%
| **Technical issue** | 1,398.0 | 20.63% | 349.0 | 20.60%
| **Cancellation request** | 1,356.0 | 20.01% | 339.0 | 20.01%
| **Product inquiry** | 1,313.0 | 19.38% | 328.0 | 19.36%
| **Billing inquiry** | 1,307.0 | 19.29% | 327.0 | 19.30%

---

## 6. Target B: Ticket Priority Class Distribution

* **Priority Training Rows**: `6,775`
* **Priority Testing Rows**: `1,694`

### Train vs. Test Class Comparison:

| Priority Class | Train Count | Train (%) | Test Count | Test (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Medium** | 1,754.0 | 25.89% | 438.0 | 25.86%
| **Critical** | 1,703.0 | 25.14% | 426.0 | 25.15%
| **High** | 1,668.0 | 24.62% | 417.0 | 24.62%
| **Low** | 1,650.0 | 24.35% | 413.0 | 24.38%

---

## 7. Vectorization & Data Leakage Protection Policy

**IMPORTANT**: TF-IDF Vectorization was **NOT** fitted during this stage.

* **Rationale**: Fitting TF-IDF on the combined dataset (or test set) would allow vocabulary frequencies and IDF statistics from the test set to leak into the training process.
* **Stage 5 Protocol**:
  1. Fit `TfidfVectorizer` **ONLY** on `category_train.csv` / `priority_train.csv`.
  2. Transform training text using the fitted vectorizer.
  3. Transform test text using the already-fitted vectorizer (with zero refitting).

---

## 8. Output Datasets & Verification

The following prepared datasets have been exported to `data/processed/`:
1. `category_train.csv` (`ticket_text`, `Ticket Type`)
2. `category_test.csv` (`ticket_text`, `Ticket Type`)
3. `priority_train.csv` (`ticket_text`, `Ticket Priority`)
4. `priority_test.csv` (`ticket_text`, `Ticket Priority`)

**Status: DATA IS 100% READY FOR SUPERVISED MODEL TRAINING (STAGE 5).**
