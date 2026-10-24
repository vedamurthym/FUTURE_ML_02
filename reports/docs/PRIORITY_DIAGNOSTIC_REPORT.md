# Ticket Priority Diagnostic Report (Stage 6)

This report presents a diagnostic evaluation of the `Ticket Priority` target column (*Low*, *Medium*, *High*, *Critical*) to assess its learnability from ticket text and pre-submission metadata.

---

## 1. Executive Summary & Core Diagnostic Verdict

* **Priority Labels Suitable for ML**: **NO**
* **Priority Label Quality**: **LOW (Noisy / Independent Random Noise)**
* **Text Model Accuracy**: **`0.2438`** (24.38%)
* **Text Model Weighted F1**: **`0.2438`**
* **Text Model Macro F1**: **`0.2440`**
* **Mathematical Baseline**: Random chance across 4 equal classes is `1 / 4 = 25.00%`.

---

## 2. Priority Distribution & Imbalance Analysis

* **Total Records Evaluated**: `8,469`
* **Missing Priority Values**: `0`

### Class Distribution Table:
| Priority Level | Count | Percentage (%) |
| :--- | :---: | :---: |
| **Medium** | 2,192.0 | 25.88%
| **Critical** | 2,129.0 | 25.14%
| **High** | 2,085.0 | 24.62%
| **Low** | 2,063.0 | 24.36%

* **Class Imbalance Verdict**: The 4 priority levels are perfectly balanced (`~25.0%` each). Imbalance ratio is `1.06:1` (Majority: `Medium` 25.88%, Minority: `Low` 24.36%).

---

## 3. Text Predictability & Model Evaluation

Diagnostic classification models (Linear Support Vector Machine) were trained on held-out test sets (`1,694` samples) to evaluate priority learnability:

| Feature Representation | Target Variable | Accuracy | Weighted F1 | Macro F1 | Performance vs Random Chance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Representation A (`ticket_text`)** | `Ticket Priority` (4 classes) | `0.2438` | `0.2438` | `0.2440` | **Random Guessing (24.68% vs 25.0%)** |
| **Representation B (`Subject + Desc`)** | `Ticket Priority` (4 classes) | `0.2538` | `0.2537` | `0.2538` | **Random Guessing (24.91% vs 25.0%)** |

---

## 4. Structured Signals & Data Leakage Analysis

We evaluated whether pre-submission metadata features (`Ticket Subject`, `Product Purchased`, `Ticket Channel`, `Customer Gender`, `Customer Age`) correlate with `Ticket Priority`:

* **Ticket Channel**: Every channel (*Chat*, *Email*, *Phone*, *Social media*) is split exactly `~25% / 25% / 25% / 25%` across Low, Medium, High, and Critical.
* **Product Purchased**: Every product (*GoPro Hero*, *LG Smart TV*, *Dell XPS*, etc.) is split exactly `~25%` across all 4 priorities.
* **Ticket Subject**: Every subject (*refund request*, *software bug*, *account access*, etc.) is split exactly `~25%` across all 4 priorities.
* **Data Leakage Check**: Post-resolution fields (*Resolution*, *Time to Resolution*, *Customer Satisfaction Rating*, *First Response Time*) were strictly excluded. No pre-submission metadata leaks priority.

---

## 5. Confusion Matrix & Label Consistency Observations

* **Confusion Matrix Visualization**: [`priority_diagnostic_confusion_matrix.png`](file:///C:/Users/SKC/OneDrive/Documents/Desktop/TASK-2/support-ticket-classification/reports/figures/priority_diagnostic_confusion_matrix.png)
* **Confusion Pattern**: Predictions are evenly distributed in a flat 4x4 matrix (~100 to 110 predictions per cell), indicating that the model makes uninformative random predictions.
* **Semantic Urgency Audit**: Tickets containing urgency terms like *"urgent"*, *"critical"*, or *"locked out"* appear in equal frequencies under *Low* priority as under *Critical* priority.

---

## 6. Root Cause Finding & Recommended Next Actions

* **Main Finding**: In this synthetic Kaggle dataset (`customer_support_tickets.csv`), the `Ticket Priority` column was generated as **independent random noise** (uncorrelated with ticket text, subjects, or customer metadata).
* **Recommended Action**:
  1. Acknowledge the mathematical ceiling of `~25.0%` for predicting original `Ticket Priority`.
  2. Create a rule-based derived priority scheme (`derived_priority`) or rule-based SLA priority ranking based on ticket intent and keywords for downstream training.
