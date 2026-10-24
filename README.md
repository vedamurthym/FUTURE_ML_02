# 🎫 AI Customer Support Ticket Classification System

An end-to-end Machine Learning and Natural Language Processing (NLP) system that automatically categorizes incoming customer support tickets and predicts their urgency/priority in real time.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Directory Structure](#-directory-structure)
- [Quick Start](#-quick-start)
- [End-to-End ML Pipeline Architecture](#-end-to-end-ml-pipeline-architecture)
- [Model Performance Summary](#-model-performance-summary)
- [Python Inference API](#-python-inference-api)
- [Flask Web Application](#-flask-web-application)
- [Key Insights & Diagnostic Discoveries](#-key-insights--diagnostic-discoveries)
- [Developer Guide for Next Coders](#-developer-guide-for-next-coders)

---

## 🚀 Project Overview

Customer support teams face high ticket volumes across diverse channels (Email, Chat, Social Media). Manual triage causes delays and inconsistent ticket routing.

This repository provides:
1. **Automated Category Classification**: Routes tickets into 7 core functional domains (*Technical Issue*, *Billing & Payment*, *Account & Access*, *Refund Request*, *Cancellation Request*, *Delivery & Order Inquiry*, *Product Inquiry*).
2. **Automated Priority Triage**: Assigns urgency ratings (*Critical*, *High*, *Medium*, *Low*) to prioritize urgent issues and minimize SLA breaches.
3. **Interactive Web App**: A production-ready Flask web interface (HTML/CSS/JavaScript) featuring single-ticket triage, bulk CSV analysis, confidence scoring, and routing recommendations.
4. **Reproducible ML Pipeline**: Modular Python scripts numbered from `01` to `08`, managed via a centralized orchestrator (`run_pipeline.py`).

---

## 📁 Directory Structure

```text
TASK-2/
└── support-ticket-classification/
    ├── app.py                     # Flask web app (HTML/CSS/JS frontend)
    ├── requirements.txt           # Python dependencies
    ├── run_pipeline.py            # Centralized pipeline runner CLI
    ├── README.md                  # Project documentation
    ├── .gitignore                 # Git ignore rules
    │
    ├── data/
    │   ├── raw/                   # Immutable raw source dataset
    │   │   └── customer_support_tickets.csv
    │   └── processed/             # Cleaned, labeled, and train/test split datasets
    │       ├── customer_support_tickets_cleaned.csv
    │       ├── customer_support_tickets_labeled.csv
    │       ├── customer_support_tickets_priority_labeled.csv
    │       ├── derived_category_train.csv
    │       ├── derived_category_test.csv
    │       ├── derived_priority_train.csv
    │       └── derived_priority_test.csv
    │
    ├── models/                    # Serialized production models & TF-IDF vectorizers
    │   ├── derived_category_model.pkl
    │   ├── derived_category_tfidf_vectorizer.pkl
    │   ├── priority_model.pkl
    │   └── priority_tfidf_vectorizer.pkl
    │
    ├── reports/                   # Audit reports, evaluation metrics, and validation logs
    │   ├── docs/                  # 📄 Markdown analytical reports & stage audits
    │   │   ├── DATA_UNDERSTANDING_REPORT.md
    │   │   ├── DATA_CLEANING_REPORT.md
    │   │   ├── EDA_REPORT.md
    │   │   ├── ML_DATA_PREPARATION_REPORT.md
    │   │   ├── CATEGORY_MODEL_DIAGNOSTIC_REPORT.md
    │   │   ├── DERIVED_CATEGORY_LABELING_REPORT.md
    │   │   ├── CATEGORY_LABEL_VALIDATION_REPORT.md
    │   │   ├── DERIVED_CATEGORY_MODEL_REPORT.md
    │   │   ├── PRIORITY_DIAGNOSTIC_REPORT.md
    │   │   ├── DERIVED_PRIORITY_LABELING_REPORT.md
    │   │   ├── PRIORITY_LABEL_VALIDATION_REPORT.md
    │   │   ├── PRIORITY_MODEL_REPORT.md
    │   │   ├── FINAL_MODEL_EVALUATION_REPORT.md
    │   │   └── WEB_APP_REPORT.md
    │   │
    │   ├── tables/                # 📊 CSV audit logs, misclassifications, and metrics
    │   │   ├── FINAL_MODEL_METRICS.csv
    │   │   ├── critical_priority_errors.csv
    │   │   ├── priority_misclassified_examples.csv
    │   │   ├── priority_model_comparison.csv
    │   │   ├── priority_label_validation_samples.csv
    │   │   ├── sample_ticket_predictions.csv
    │   │   └── ... (24 evaluation tables)
    │   │
    │   └── figures/               # 🖼️ High-resolution charts & confusion matrices
    │       ├── category_vs_priority.png
    │       ├── ticket_category_distribution.png
    │       ├── ticket_priority_distribution.png
    │       ├── ticket_text_length_distribution.png
    │       ├── top_20_words.png
    │       ├── final_category_confusion_matrix.png
    │       └── final_priority_confusion_matrix.png
    │
    └── src/                       # Modular pipeline source code
        ├── 01_data_understanding.py
        ├── 02_data_cleaning.py
        ├── 03_eda.py
        ├── 04_prepare_data.py
        ├── 05_train_category_model.py
        ├── 05a_diagnose_category_model.py
        ├── 05b_create_category_labels.py
        ├── 05c_validate_category_labels.py
        ├── 05d_train_category_model.py
        ├── 06_priority_diagnosis.py
        ├── 06b_create_priority_labels.py
        ├── 06c_validate_priority_labels.py
        ├── 06d_train_priority_model.py
        ├── 07_final_model_evaluation.py
        └── 08_predict_ticket.py
```

---

## ⚡ Quick Start

### 1. Environment Setup
Clone the repository and install required packages:
```bash
cd support-ticket-classification
pip install -r requirements.txt
```

### 2. Launch the Flask Web Application
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

### 3. Run Pipeline Stages via Orchestrator
List available stages:
```bash
python run_pipeline.py --list
```

Run a specific stage (e.g., Inference smoke test):
```bash
python run_pipeline.py --stage 08
```

Run the entire pipeline from scratch:
```bash
python run_pipeline.py --all
```

---

## 🔄 End-to-End ML Pipeline Architecture

The pipeline is organized sequentially from data ingestion to model deployment:

| Stage | Script | Purpose | Key Artifacts |
|:---|:---|:---|:---|
| **01** | [`src/01_data_understanding.py`](file:///src/01_data_understanding.py) | Inspect raw data schema, missing values, duplicates, and cardinality. | `reports/DATA_UNDERSTANDING_REPORT.md` |
| **02** | [`src/02_data_cleaning.py`](file:///src/02_data_cleaning.py) | Clean text, remove boilerplate/noise, handle missing fields, standardize columns. | `data/processed/customer_support_tickets_cleaned.csv` |
| **03** | [`src/03_eda.py`](file:///src/03_eda.py) | Generate statistical distributions, n-grams, word frequency plots, and correlations. | `reports/figures/*.png`, `reports/EDA_REPORT.md` |
| **04** | [`src/04_prepare_data.py`](file:///src/04_prepare_data.py) | Build stratified train-test splits (80/20) and TF-IDF feature pipelines. | `data/processed/*_train.csv`, `*_test.csv` |
| **05a** | [`src/05a_diagnose_category_model.py`](file:///src/05a_diagnose_category_model.py) | Diagnose label noise in original raw categories (accuracy plateaued near ~20%). | `reports/CATEGORY_MODEL_DIAGNOSTIC_REPORT.md` |
| **05b** | [`src/05b_create_category_labels.py`](file:///src/05b_create_category_labels.py) | Generate deterministic, rule-based semantic category ground truth (7 classes). | `data/processed/customer_support_tickets_labeled.csv` |
| **05c** | [`src/05c_validate_category_labels.py`](file:///src/05c_validate_category_labels.py) | Validate mutual exclusivity and semantic consistency of category rules. | `reports/CATEGORY_LABEL_VALIDATION_REPORT.md` |
| **05d** | [`src/05d_train_category_model.py`](file:///src/05d_train_category_model.py) | Train and tune high-accuracy LinearSVC category classifier with TF-IDF. | `models/derived_category_model.pkl` |
| **06** | [`src/06_priority_diagnosis.py`](file:///src/06_priority_diagnosis.py) | Diagnose raw priority labels; proves ~24.7% accuracy (pure random chance across 4 classes). | `reports/PRIORITY_DIAGNOSTIC_REPORT.md` |
| **06b** | [`src/06b_create_priority_labels.py`](file:///src/06b_create_priority_labels.py) | Apply deterministic urgency heuristics (*Critical*, *High*, *Medium*, *Low*). | `data/processed/customer_support_tickets_priority_labeled.csv` |
| **06c** | [`src/06c_validate_priority_labels.py`](file:///src/06c_validate_priority_labels.py) | Validate priority distribution and flag potential edge cases for human review. | `reports/PRIORITY_LABEL_VALIDATION_REPORT.md` |
| **06d** | [`src/06d_train_priority_model.py`](file:///src/06d_train_priority_model.py) | Train balanced LinearSVC priority classifier with balanced class weighting. | `models/priority_model.pkl` |
| **07** | [`src/07_final_model_evaluation.py`](file:///src/07_final_model_evaluation.py) | Comprehensive multi-metric benchmark on held-out test sets. | `reports/FINAL_MODEL_EVALUATION_REPORT.md` |
| **08** | [`src/08_predict_ticket.py`](file:///src/08_predict_ticket.py) | Unified inference engine supporting single-ticket and batch ticket classification. | `reports/sample_ticket_predictions.csv` |

---

## 📊 Model Performance Summary

Evaluated on held-out test sets ($N = 1,694$ tickets per test set):

| Task | Final Model | Accuracy | Macro F1 | Weighted F1 | Notes |
|:---|:---|:---:|:---:|:---:|:---|
| **Category Classification** | LinearSVC (TF-IDF 1-2 Grams) | **98.70%** | **0.9856** | **0.9870** | Strong separation across all 7 ticket domains |
| **Priority Classification** | LinearSVC (Class Weighted) | **86.78%** | **0.7818** | **0.8660** | Prioritizes Critical and High urgency issues |

*Detailed per-class metrics and confusion matrices can be found in [`reports/FINAL_MODEL_EVALUATION_REPORT.md`](file:///reports/FINAL_MODEL_EVALUATION_REPORT.md).*

---

## 💻 Python Inference API

You can easily integrate the prediction pipeline into existing scripts, backends, or microservices:

```python
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.append(str(Path(".").resolve()))

from src import predict_ticket_module  # or import dynamically via src.08_predict_ticket
import importlib
predict_mod = importlib.import_module("src.08_predict_ticket")

# 1. Single Ticket Prediction
result = predict_mod.predict_ticket(
    "My server crashed and our entire billing portal is throwing 500 errors! Urgent help needed!"
)

print(f"Category: {result['category']} (Confidence Score: {result['category_confidence']:.2f})")
print(f"Priority: {result['priority']} (Confidence Score: {result['priority_confidence']:.2f})")

# 2. Batch Ticket Prediction (Pandas DataFrame)
import pandas as pd
df = pd.DataFrame({
    "Ticket Description": [
        "I need a refund for my order #12345.",
        "Unable to reset my password on the mobile app."
    ]
})
predictions_df = predict_mod.predict_tickets(df, text_col="Ticket Description")
print(predictions_df[["Ticket Description", "predicted_category", "predicted_priority"]])
```

---

## 🌐 Flask Web Application

The repository includes a Flask web interface ([`app.py`](file:///app.py)) with a fully custom **HTML/CSS/JavaScript** frontend:

### Key Features:
- **Real-Time Triage Mode**: Type or paste any ticket description, select quick test presets, and get instant category & priority predictions with visual priority badges and routing instructions.
- **Batch CSV Analysis Mode**: Upload a CSV file via drag-and-drop, auto-detect text columns, run high-throughput batch classification, and download enriched CSV results.
- **System Metrics & Architecture Viewer**: View model benchmark metrics and performance statistics directly from the UI.

### Run the App:
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

---

## 🔍 Key Insights & Diagnostic Discoveries

1. **Synthetic Noise in Raw Dataset**:
   - The initial raw `Ticket Type` and `Ticket Priority` columns exhibited near-random uniform distributions across classes.
   - Initial models scored $\sim 20-25\%$ test accuracy (equivalent to random guessing).
2. **Root Cause Analysis**:
   - In the raw dataset, ticket descriptions (e.g. *"I want to cancel my subscription"*) were randomly paired with conflicting labels (e.g. *Product Inquiry* and *Low Priority*).
3. **Derived Ground Truth Resolution**:
   - We implemented deterministic, auditable semantic rule engines in Stages `05b` and `06b` to extract genuine intent and urgency from ticket text.
   - Classification accuracy surged to **98.7%** (Category) and **86.8%** (Priority) on held-out evaluation sets, creating a viable triage model.

---

## 🛠️ Developer Guide for Next Coders

### Adding New Classification Categories
1. Update keyword definitions in [`src/05b_create_category_labels.py`](file:///src/05b_create_category_labels.py).
2. Run validation via `python run_pipeline.py --stage 05c`.
3. Retrain category model: `python run_pipeline.py --stage 05d`.
4. Re-evaluate final performance: `python run_pipeline.py --stage 07`.

### Modifying Priority / Urgency Rules
1. Edit urgency heuristics in [`src/06b_create_priority_labels.py`](file:///src/06b_create_priority_labels.py).
2. Validate distribution via `python run_pipeline.py --stage 06c`.
3. Retrain priority model: `python run_pipeline.py --stage 06d`.
4. Run inference smoke tests: `python run_pipeline.py --stage 08`.

### Code Conventions
- Use `Path(__file__).resolve().parent` for dynamic path resolution so scripts run reliably regardless of working directory.
- Preserve deterministic random seeds (`random_state=42`) across train-test splits and estimators for reproducibility.
- Always save evaluation outputs into `reports/` and figures into `reports/figures/`.
