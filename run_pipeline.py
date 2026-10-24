"""
Support Ticket Classification - End-to-End Pipeline Orchestrator
================================================================
A centralized runner to execute stages of the machine learning pipeline
sequentially or individually.

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --stage 01
  python run_pipeline.py --stage 08
  python run_pipeline.py --list
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# Pipeline stages in canonical execution order
PIPELINE_STAGES = [
    {
        "id": "01",
        "script": "01_data_understanding.py",
        "name": "Data Understanding & Baseline Inspection",
        "description": "Loads raw customer support dataset and audits schemas, nulls, and distributions.",
        "outputs": ["reports/docs/DATA_UNDERSTANDING_REPORT.md"],
    },
    {
        "id": "02",
        "script": "02_data_cleaning.py",
        "name": "Data Cleaning & Preprocessing",
        "description": "Standardizes text columns, removes boilerplate, and handles missing values.",
        "outputs": ["data/processed/customer_support_tickets_cleaned.csv", "reports/docs/DATA_CLEANING_REPORT.md"],
    },
    {
        "id": "03",
        "script": "03_eda.py",
        "name": "Exploratory Data Analysis (EDA)",
        "description": "Generates distributions, class balances, ngram frequencies, and visual charts.",
        "outputs": ["reports/docs/EDA_REPORT.md", "reports/figures/ticket_category_distribution.png"],
    },
    {
        "id": "04",
        "script": "04_prepare_data.py",
        "name": "Data Preparation & Train-Test Splits",
        "description": "Builds train/test splits for original category and priority target baselines.",
        "outputs": ["data/processed/category_train.csv", "reports/docs/ML_DATA_PREPARATION_REPORT.md"],
    },
    {
        "id": "05",
        "script": "05_train_category_model.py",
        "name": "Baseline Category Model Training",
        "description": "Trains initial category classifiers (Logistic Regression, LinearSVC, Naive Bayes).",
        "outputs": ["models/category_model.pkl", "reports/docs/CATEGORY_MODEL_REPORT.md"],
    },
    {
        "id": "05a",
        "script": "05a_diagnose_category_model.py",
        "name": "Category Label Quality Diagnosis",
        "description": "Diagnoses noise and semantic misalignment in original Ticket Type labels.",
        "outputs": ["reports/docs/CATEGORY_MODEL_DIAGNOSTIC_REPORT.md", "reports/tables/category_prediction_samples.csv"],
    },
    {
        "id": "05b",
        "script": "05b_create_category_labels.py",
        "name": "Derived Category Label Generation",
        "description": "Applies deterministic, rule-based semantic category labeling.",
        "outputs": ["data/processed/customer_support_tickets_labeled.csv", "reports/docs/DERIVED_CATEGORY_LABELING_REPORT.md", "reports/tables/derived_category_examples.csv"],
    },
    {
        "id": "05c",
        "script": "05c_validate_category_labels.py",
        "name": "Derived Category Label Validation",
        "description": "Validates keyword coverage, mutual exclusivity, and class distribution.",
        "outputs": ["reports/docs/CATEGORY_LABEL_VALIDATION_REPORT.md", "reports/tables/category_label_validation_samples.csv"],
    },
    {
        "id": "05d",
        "script": "05d_train_category_model.py",
        "name": "Derived Category Model Training",
        "description": "Trains final high-accuracy category classifier on derived ground truth.",
        "outputs": ["models/derived_category_model.pkl", "reports/docs/DERIVED_CATEGORY_MODEL_REPORT.md", "reports/tables/category_derived_model_comparison.csv"],
    },
    {
        "id": "06",
        "script": "06_priority_diagnosis.py",
        "name": "Priority Label Quality Diagnosis",
        "description": "Demonstrates that original Ticket Priority is random synthetic noise (~25% accuracy).",
        "outputs": ["reports/docs/PRIORITY_DIAGNOSTIC_REPORT.md", "reports/tables/priority_label_examples.csv"],
    },
    {
        "id": "06b",
        "script": "06b_create_priority_labels.py",
        "name": "Derived Priority Label Generation",
        "description": "Applies rule-based urgency heuristics (Critical, High, Medium, Low).",
        "outputs": ["data/processed/customer_support_tickets_priority_labeled.csv", "reports/docs/DERIVED_PRIORITY_LABELING_REPORT.md", "reports/tables/priority_labeling_rules.csv"],
    },
    {
        "id": "06c",
        "script": "06c_validate_priority_labels.py",
        "name": "Derived Priority Label Validation",
        "description": "Validates priority rules and flags edge cases for human review.",
        "outputs": ["reports/docs/PRIORITY_LABEL_VALIDATION_REPORT.md", "reports/tables/priority_label_validation_samples.csv"],
    },
    {
        "id": "06d",
        "script": "06d_train_priority_model.py",
        "name": "Derived Priority Model Training",
        "description": "Trains and tunes final priority classifier with class weighting.",
        "outputs": ["models/priority_model.pkl", "reports/docs/PRIORITY_MODEL_REPORT.md", "reports/tables/priority_model_comparison.csv"],
    },
    {
        "id": "07",
        "script": "07_final_model_evaluation.py",
        "name": "Final Model Evaluation & Benchmark",
        "description": "Evaluates both final models on held-out test sets and generates unified metrics.",
        "outputs": ["reports/docs/FINAL_MODEL_EVALUATION_REPORT.md", "reports/tables/FINAL_MODEL_METRICS.csv"],
    },
    {
        "id": "08",
        "script": "08_predict_ticket.py",
        "name": "Unified Ticket Prediction & Inference Pipeline",
        "description": "Runs end-to-end inference smoke tests and batch predictions for new tickets.",
        "outputs": ["reports/tables/sample_ticket_predictions.csv"],
    },
]


def list_stages():
    """Prints all pipeline stages."""
    print("\n" + "=" * 80)
    print("SUPPORT TICKET CLASSIFICATION PIPELINE STAGES")
    print("=" * 80)
    for stage in PIPELINE_STAGES:
        print(f"[{stage['id']:<4}] {stage['script']:<32} - {stage['name']}")
        print(f"       {stage['description']}")
    print("=" * 80)
    print("\nRun a specific stage:  python run_pipeline.py --stage <id_or_script_name>")
    print("Run all stages:        python run_pipeline.py --all\n")


def run_stage(stage_info: dict) -> bool:
    """Executes a single pipeline stage script."""
    script_path = SRC_DIR / stage_info["script"]
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    print("\n" + "=" * 80)
    print(f"RUNNING STAGE {stage_info['id']}: {stage_info['name']}")
    print(f"Script: src/{stage_info['script']}")
    print("=" * 80)

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n--> Stage {stage_info['id']} completed successfully in {elapsed:.2f}s.")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[FAILED] Stage {stage_info['id']} exited with code {e.returncode} after {elapsed:.2f}s.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end Support Ticket Classification pipeline stages."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the entire pipeline from Stage 01 through Stage 08 in order."
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Stage identifier (e.g., '01', '05b', '08') or script name (e.g., '08_predict_ticket.py')."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available pipeline stages and exit."
    )

    args = parser.parse_args()

    if args.list or (not args.all and not args.stage):
        list_stages()
        return

    if args.all:
        print("\nStarting full pipeline execution (Stages 01 -> 08)...")
        overall_start = time.time()
        for stage in PIPELINE_STAGES:
            success = run_stage(stage)
            if not success:
                print(f"\n[ABORTED] Pipeline execution stopped due to failure in stage {stage['id']}.")
                sys.exit(1)
        total_time = time.time() - overall_start
        print("\n" + "=" * 80)
        print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY in {total_time:.2f}s!")
        print("=" * 80)
        return

    if args.stage:
        target = args.stage.strip().lower()
        matched_stage = None
        for stage in PIPELINE_STAGES:
            if stage["id"].lower() == target or stage["script"].lower() == target or stage["script"].lower().startswith(target):
                matched_stage = stage
                break

        if not matched_stage:
            print(f"[ERROR] Stage '{args.stage}' not found. Use --list to view valid stages.")
            sys.exit(1)

        success = run_stage(matched_stage)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
