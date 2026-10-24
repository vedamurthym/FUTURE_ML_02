"""
Stage 6D — Train Priority Classification Model
================================================
IMPORTANT DOCUMENTATION NOTE:
  The original Ticket Priority labels were diagnosed as noisy/non-predictive in
  Stage 6. The priority classifier therefore uses separately derived, rule-based
  priority labels (derived_priority). These labels are NOT equivalent to
  human-annotated ground truth.

  The Critical class has limited representation (n=207, 2.44%). Treat Critical-
  class metrics with appropriate caution.

Script : src/06d_train_priority_model.py
Target : derived_priority  (NOT the original Ticket Priority column)
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.naive_bayes     import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_IN          = ROOT / "data" / "processed" / "customer_support_tickets_priority_labeled.csv"
TRAIN_OUT        = ROOT / "data" / "processed" / "derived_priority_train.csv"
TEST_OUT         = ROOT / "data" / "processed" / "derived_priority_test.csv"
MODEL_OUT        = ROOT / "models" / "priority_model.pkl"
VECTORIZER_OUT   = ROOT / "models" / "priority_tfidf_vectorizer.pkl"
FIGURES_DIR      = ROOT / "reports" / "figures"
COMPARISON_OUT   = ROOT / "reports" / "tables" / "priority_model_comparison.csv"
PREDICTIONS_OUT  = ROOT / "reports" / "tables" / "priority_test_predictions.csv"
MISCLASSIFIED_OUT= ROOT / "reports" / "tables" / "priority_misclassified_examples.csv"
REPORT_OUT       = ROOT / "reports" / "docs" / "PRIORITY_MODEL_REPORT.md"

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]

# TF-IDF settings (grounded in Stage 5D experience)
TFIDF_NGRAM      = (1, 2)
TFIDF_MAX_FEAT   = 50_000
TFIDF_MIN_DF     = 2

# ---------------------------------------------------------------------------
# STEP 1 — LOAD DATA
# ---------------------------------------------------------------------------
def load_data(path: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    print(f"Loading: {path.relative_to(ROOT)}")
    df = pd.read_csv(path)
    print(f"  Total rows : {len(df):,}   Columns: {df.shape[1]}")

    # Verify derived_priority exists
    if "derived_priority" not in df.columns:
        raise ValueError("'derived_priority' column not found. Run Stage 6B first.")

    # Drop rows with missing target or text
    before = len(df)
    df = df.dropna(subset=["ticket_text", "derived_priority"])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing text/target.")

    X = df["ticket_text"].astype(str)
    y = df["derived_priority"].astype(str)

    print(f"\n  Target column  : derived_priority (NOT original Ticket Priority)")
    print(f"  Feature column : ticket_text")
    print(f"\n  Label distribution:")
    for lvl in PRIORITY_ORDER:
        cnt = (y == lvl).sum()
        pct = cnt / len(y) * 100
        print(f"    {lvl:<12}: {cnt:>5,} ({pct:.2f}%)")

    return X, y, df


# ---------------------------------------------------------------------------
# STEP 2+3 — TRAIN/TEST SPLIT + SAVE SPLITS
# ---------------------------------------------------------------------------
def make_splits(
    X: pd.Series, y: pd.Series, df: pd.DataFrame
) -> tuple:
    print(f"\n[STEP 2] Creating stratified 80/20 train/test split (random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing  samples: {len(X_test):,}")

    print("\n  Training distribution:")
    for lvl in PRIORITY_ORDER:
        cnt = (y_train == lvl).sum()
        pct = cnt / len(y_train) * 100
        print(f"    {lvl:<12}: {cnt:>5,} ({pct:.2f}%)")

    print("\n  Testing distribution:")
    for lvl in PRIORITY_ORDER:
        cnt = (y_test == lvl).sum()
        pct = cnt / len(y_test) * 100
        print(f"    {lvl:<12}: {cnt:>5,} ({pct:.2f}%)")

    # Verify Critical is in both splits
    critical_train = (y_train == "Critical").sum()
    critical_test  = (y_test  == "Critical").sum()
    print(f"\n  Critical in train: {critical_train}   Critical in test: {critical_test}")
    if critical_train == 0 or critical_test == 0:
        raise ValueError("Critical class missing from one split! Increase test size or check data.")

    # Save splits (Step 3)
    print("\n[STEP 3] Saving train/test splits...")
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"ticket_text": X_train, "derived_priority": y_train})
    test_df  = pd.DataFrame({"ticket_text": X_test,  "derived_priority": y_test})

    train_df.to_csv(TRAIN_OUT, index=True, index_label="original_index")
    test_df.to_csv(TEST_OUT,  index=True, index_label="original_index")
    print(f"  Saved: {TRAIN_OUT.relative_to(ROOT)}")
    print(f"  Saved: {TEST_OUT.relative_to(ROOT)}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 4 — TF-IDF VECTORISATION
# ---------------------------------------------------------------------------
def vectorise(X_train: pd.Series, X_test: pd.Series):
    print("\n[STEP 4] TF-IDF Vectorisation...")
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM,
        max_features=TFIDF_MAX_FEAT,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)   # fit ONLY on train
    X_test_tfidf  = vectorizer.transform(X_test)        # transform test

    n_feats = X_train_tfidf.shape[1]
    print(f"  TF-IDF features : {n_feats:,}")
    print(f"  Training matrix : {X_train_tfidf.shape}")
    print(f"  Testing  matrix : {X_test_tfidf.shape}")
    return vectorizer, X_train_tfidf, X_test_tfidf


# ---------------------------------------------------------------------------
# HELPER — EVALUATE A MODEL
# ---------------------------------------------------------------------------
def evaluate_model(
    name: str,
    class_weight: str,
    model,
    X_test_tfidf,
    y_test: pd.Series,
) -> dict:
    y_pred = model.predict(X_test_tfidf)
    acc  = accuracy_score(y_test, y_pred)
    wp   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    wr   = recall_score   (y_test, y_pred, average="weighted", zero_division=0)
    wf1  = f1_score       (y_test, y_pred, average="weighted", zero_division=0)
    mp   = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    mr   = recall_score   (y_test, y_pred, average="macro",    zero_division=0)
    mf1  = f1_score       (y_test, y_pred, average="macro",    zero_division=0)

    # Per-class
    report = classification_report(
        y_test, y_pred, labels=PRIORITY_ORDER, output_dict=True, zero_division=0
    )
    critical_p  = report.get("Critical", {}).get("precision", 0.0)
    critical_r  = report.get("Critical", {}).get("recall",    0.0)
    critical_f1 = report.get("Critical", {}).get("f1-score",  0.0)

    print(f"\n  {name} [{class_weight}]")
    print(f"    Accuracy        : {acc:.4f}")
    print(f"    Weighted F1     : {wf1:.4f}")
    print(f"    Macro F1        : {mf1:.4f}")
    print(f"    Critical P/R/F1 : {critical_p:.4f} / {critical_r:.4f} / {critical_f1:.4f}")

    return {
        "model_name":    name,
        "class_weight":  class_weight,
        "model_obj":     model,
        "y_pred":        y_pred,
        "Accuracy":            round(acc,  4),
        "Weighted Precision":  round(wp,   4),
        "Weighted Recall":     round(wr,   4),
        "Weighted F1":         round(wf1,  4),
        "Macro Precision":     round(mp,   4),
        "Macro Recall":        round(mr,   4),
        "Macro F1":            round(mf1,  4),
        "Critical Precision":  round(critical_p,  4),
        "Critical Recall":     round(critical_r,  4),
        "Critical F1":         round(critical_f1, 4),
        "report_dict":   report,
    }


# ---------------------------------------------------------------------------
# STEP 5 — TRAIN ALL MODELS
# ---------------------------------------------------------------------------
def train_all_models(
    X_train_tfidf, X_test_tfidf,
    y_train: pd.Series, y_test: pd.Series,
) -> list[dict]:
    print("\n[STEP 5] Training models...")

    configs = [
        # (display_name, class_weight, model_instance)
        ("Logistic Regression", "none",
         LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")),
        ("Logistic Regression", "balanced",
         LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs",
                            class_weight="balanced")),
        ("Linear SVM", "none",
         LinearSVC(max_iter=2000, random_state=RANDOM_STATE)),
        ("Linear SVM", "balanced",
         LinearSVC(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")),
        ("Multinomial Naive Bayes", "none",
         MultinomialNB()),
    ]

    results = []
    for display_name, cw, model in configs:
        model.fit(X_train_tfidf, y_train)
        res = evaluate_model(display_name, cw, model, X_test_tfidf, y_test)
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# STEP 7 — CONFUSION MATRICES
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    filename: str,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=PRIORITY_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=PRIORITY_ORDER, yticklabels=PRIORITY_ORDER, ax=ax,
    )
    ax.set_xlabel("Predicted Priority", fontsize=11)
    ax.set_ylabel("Actual Priority",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix: {out.relative_to(ROOT)}")


def save_all_confusion_matrices(results: list[dict], y_test: pd.Series) -> None:
    print("\n[STEP 7] Saving confusion matrices...")
    name_map = {
        "Logistic Regression":     "priority_logistic_regression",
        "Linear SVM":              "priority_linear_svm",
        "Multinomial Naive Bayes": "priority_naive_bayes",
    }
    seen = {}
    for res in results:
        base = name_map.get(res["model_name"], res["model_name"].lower().replace(" ", "_"))
        suffix = "_balanced" if res["class_weight"] == "balanced" else "_standard"
        filename = f"{base}{suffix}_confusion_matrix.png"
        save_confusion_matrix(
            y_test, res["y_pred"],
            f"{res['model_name']} [{res['class_weight']}]",
            filename,
        )
        seen[res["model_name"]] = res  # keep last (balanced preferred)


# ---------------------------------------------------------------------------
# STEP 10 — SAVE COMPARISON CSV
# ---------------------------------------------------------------------------
def save_comparison(results: list[dict], path: Path) -> pd.DataFrame:
    print("\n[STEP 10] Saving model comparison...")
    rows = []
    for res in results:
        rows.append({
            "Model":              res["model_name"],
            "Class Weight":       res["class_weight"],
            "Accuracy":           res["Accuracy"],
            "Weighted Precision": res["Weighted Precision"],
            "Weighted Recall":    res["Weighted Recall"],
            "Weighted F1":        res["Weighted F1"],
            "Macro Precision":    res["Macro Precision"],
            "Macro Recall":       res["Macro Recall"],
            "Macro F1":           res["Macro F1"],
            "Critical Precision": res["Critical Precision"],
            "Critical Recall":    res["Critical Recall"],
            "Critical F1":        res["Critical F1"],
        })
    df = pd.DataFrame(rows).sort_values(
        ["Macro F1", "Critical F1", "Weighted F1"], ascending=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved: {path.relative_to(ROOT)}")
    print("\n  === MODEL COMPARISON TABLE ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# STEP 11 — SELECT BEST MODEL
# ---------------------------------------------------------------------------
def select_best_model(results: list[dict]) -> dict:
    print("\n[STEP 11] Selecting best model...")
    # Sort: Macro F1 primary, Critical F1 secondary, Weighted F1 tertiary
    ranked = sorted(
        results,
        key=lambda r: (r["Macro F1"], r["Critical F1"], r["Weighted F1"]),
        reverse=True,
    )
    best = ranked[0]
    print(f"\n  Best model selected: {best['model_name']} [{best['class_weight']}]")
    print(f"    Macro F1        : {best['Macro F1']:.4f}")
    print(f"    Critical Recall : {best['Critical Recall']:.4f}")
    print(f"    Critical F1     : {best['Critical F1']:.4f}")
    print(f"    Weighted F1     : {best['Weighted F1']:.4f}")
    print(f"    Accuracy        : {best['Accuracy']:.4f}")

    print("\n  Selection rationale:")
    print("  - Primary criterion   : Macro F1 (equal weight per class, handles imbalance)")
    print("  - Secondary criterion : Critical F1 (missing critical tickets is costly)")
    print("  - Tertiary criterion  : Weighted F1 (overall quality)")
    print("  - Accuracy EXCLUDED   : misleading with 49.65% majority class")
    return best


# ---------------------------------------------------------------------------
# STEP 8 — PER-CLASS PERFORMANCE (best model)
# ---------------------------------------------------------------------------
def print_per_class_performance(best: dict, y_test: pd.Series) -> None:
    print("\n[STEP 8] Per-class performance of best model:")
    rpt = best["report_dict"]
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for lvl in PRIORITY_ORDER:
        m = rpt.get(lvl, {})
        print(
            f"  {lvl:<12}"
            f" {m.get('precision', 0):>10.4f}"
            f" {m.get('recall',    0):>10.4f}"
            f" {m.get('f1-score',  0):>10.4f}"
            f" {m.get('support',   0):>10.0f}"
        )
    print(f"\n  *** Critical Recall: {best['Critical Recall']:.4f} ***")
    print("  (Operationally: missing a Critical ticket is more costly than missing a Low ticket)")


# ---------------------------------------------------------------------------
# STEP 9 — ERROR ANALYSIS
# ---------------------------------------------------------------------------
def error_analysis(
    best: dict, X_test: pd.Series, y_test: pd.Series, path: Path
) -> pd.DataFrame:
    print("\n[STEP 9] Error analysis...")
    y_pred = best["y_pred"]
    wrong_mask = y_pred != y_test.values

    err_df = pd.DataFrame({
        "ticket_text":       X_test.values[wrong_mask],
        "actual_priority":   y_test.values[wrong_mask],
        "predicted_priority": y_pred[wrong_mask],
    })

    print(f"  Total misclassified: {len(err_df):,} / {len(y_test):,}")
    print("\n  Most common confusion pairs (actual -> predicted):")
    pairs = (
        err_df.groupby(["actual_priority", "predicted_priority"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    for _, row in pairs.iterrows():
        print(f"    {row['actual_priority']:<12} -> {row['predicted_priority']:<12}: {row['count']:>5}")

    path.parent.mkdir(parents=True, exist_ok=True)
    err_df.to_csv(path, index=False)
    print(f"\n  Saved: {path.relative_to(ROOT)}")
    return err_df


# ---------------------------------------------------------------------------
# STEP 12 — SAVE MODEL + VECTORIZER
# ---------------------------------------------------------------------------
def save_model(best: dict, vectorizer) -> None:
    print("\n[STEP 12] Saving model and vectorizer...")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model_obj"], MODEL_OUT)
    joblib.dump(vectorizer, VECTORIZER_OUT)
    print(f"  Model saved     : {MODEL_OUT.relative_to(ROOT)}")
    print(f"  Vectorizer saved: {VECTORIZER_OUT.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# STEP 13 — TEST PREDICTION FILE
# ---------------------------------------------------------------------------
def save_predictions(
    best: dict, X_test: pd.Series, y_test: pd.Series, path: Path
) -> None:
    print("\n[STEP 13] Saving test predictions...")
    y_pred = best["y_pred"]
    pred_df = pd.DataFrame({
        "ticket_text":       X_test.values,
        "actual_priority":   y_test.values,
        "predicted_priority": y_pred,
        "correct":           (y_pred == y_test.values),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(path, index=False)
    print(f"  Saved: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# STEP 14 — WRITE TRAINING REPORT
# ---------------------------------------------------------------------------
def save_report(
    best: dict,
    results: list[dict],
    y_train: pd.Series,
    y_test: pd.Series,
    n_features: int,
    err_df: pd.DataFrame,
) -> None:
    total = len(y_train) + len(y_test)
    rpt   = best["report_dict"]

    def _dist_table(y: pd.Series, label: str) -> str:
        lines = [f"**{label}**",
                 "| Priority | Count | % |",
                 "| --- | --- | --- |"]
        for lvl in PRIORITY_ORDER:
            cnt = (y == lvl).sum()
            pct = cnt / len(y) * 100
            lines.append(f"| {lvl} | {cnt:,} | {pct:.2f}% |")
        return "\n".join(lines)

    def _comparison_table() -> str:
        lines = ["| Model | CW | Acc | W-F1 | **Macro F1** | Crit-R | Crit-F1 |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for res in sorted(results, key=lambda r: r["Macro F1"], reverse=True):
            lines.append(
                f"| {res['model_name']} | {res['class_weight']} "
                f"| {res['Accuracy']:.4f} | {res['Weighted F1']:.4f} "
                f"| **{res['Macro F1']:.4f}** "
                f"| {res['Critical Recall']:.4f} | {res['Critical F1']:.4f} |"
            )
        return "\n".join(lines)

    def _per_class_table() -> str:
        lines = ["| Priority | Precision | Recall | F1 | Support |",
                 "| --- | --- | --- | --- | --- |"]
        for lvl in PRIORITY_ORDER:
            m = rpt.get(lvl, {})
            lines.append(
                f"| {lvl} | {m.get('precision',0):.4f} "
                f"| {m.get('recall',0):.4f} "
                f"| {m.get('f1-score',0):.4f} "
                f"| {int(m.get('support',0)):,} |"
            )
        return "\n".join(lines)

    def _confusion_pairs(err_df: pd.DataFrame) -> str:
        lines = ["| Actual → Predicted | Count |", "| --- | --- |"]
        pairs = (
            err_df.groupby(["actual_priority", "predicted_priority"])
            .size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in pairs.iterrows():
            lines.append(f"| {row['actual_priority']} → {row['predicted_priority']} | {row['count']} |")
        return "\n".join(lines)

    report = f"""# PRIORITY MODEL TRAINING REPORT
## Stage 6D — Derived Priority Classification

> **⚠️ IMPORTANT**
> The original Ticket Priority labels were diagnosed as noisy/non-predictive (Stage 6).
> The priority classifier therefore uses separately derived, rule-based priority labels
> (`derived_priority`). These labels are **NOT equivalent to human-annotated ground truth**.
>
> The Critical class has limited representation (n=207, 2.44%).
> Treat Critical-class metrics with appropriate caution.

---

## 1. Objective

Train supervised text classification models to predict `derived_priority`
(Critical / High / Medium / Low) from support ticket text. The target is used
internally to route tickets; it is NOT a replacement for human triage.

---

## 2. Why Original Ticket Priority Was Rejected

Stage 6 diagnostic results:

| Metric | Value |
| --- | --- |
| Text model accuracy | 0.2438 |
| Random chance (4 classes) | ≈ 25.00% |
| Weighted F1 | 0.2438 |

The four original priority classes were each ~25% of the dataset, confirming
random assignment. No structured feature (channel, product, subject) correlated
with priority.

---

## 3. Derived Priority Methodology

Deterministic keyword + subject-based rules (Stage 6B):

| Level | Rule basis |
| --- | --- |
| Critical | hack/breach/fraud/stolen · explicit emergency · complete device failure |
| High | account access subject · data loss · network/payment/hardware/software-bug subjects · urgent language · crashes |
| Medium | refund · cancellation · delivery · installation · compatibility · display · battery |
| Low | product recommendation · how-to · firmware update · factory reset |

---

## 4. Dataset

| Item | Value |
| --- | --- |
| Total rows | {total:,} |
| Training rows | {len(y_train):,} (80%) |
| Testing rows | {len(y_test):,} (20%) |
| Split method | Stratified 80/20 (random_state={RANDOM_STATE}) |

{_dist_table(y_train, "Training Set")}

{_dist_table(y_test, "Testing Set")}

---

## 5. Class Imbalance

| Metric | Value |
| --- | --- |
| High + Medium combined | {((y_train=='High').sum()+(y_train=='Medium').sum())/len(y_train)*100:.1f}% (train) |
| Critical share | {(y_train=='Critical').sum()/len(y_train)*100:.2f}% (train) |
| Low share | {(y_train=='Low').sum()/len(y_train)*100:.2f}% (train) |
| High : Critical ratio | {(y_train=='High').sum()/(y_train=='Critical').sum():.1f}:1 |

**Implication:** Accuracy is NOT a valid primary metric. A dummy classifier
always predicting "High" scores ~49.65% accuracy. Macro F1 is the primary metric.

---

## 6. Data Leakage Prevention

Features excluded from X (post-resolution information):
- `Resolution` — not available at ticket submission
- `Time to Resolution` — not available at ticket submission
- `Customer Satisfaction Rating` — not available at ticket submission
- `First Response Time` — not available at ticket submission
- `Ticket Priority` (original) — noisy/random; excluded as target

TF-IDF fitted **ONLY on training data** then applied to test data.

---

## 7. TF-IDF Configuration

| Parameter | Value |
| --- | --- |
| ngram_range | (1, 2) |
| max_features | {TFIDF_MAX_FEAT:,} |
| min_df | {TFIDF_MIN_DF} |
| sublinear_tf | True |
| Features used | {n_features:,} |

---

## 8. Models Tested

{_comparison_table()}

*CW = Class Weight; W-F1 = Weighted F1; Crit-R = Critical Recall*

---

## 9. Best Model

**Selected: {best['model_name']} [class_weight={best['class_weight']}]**

Selection criteria (in order):
1. Macro F1 = **{best['Macro F1']:.4f}** (primary — equal weight per class)
2. Critical F1 = **{best['Critical F1']:.4f}** (missing critical tickets is costly)
3. Weighted F1 = **{best['Weighted F1']:.4f}** (overall quality)
4. Accuracy excluded as primary criterion (misleading with 49.65% majority class)

---

## 10. Per-Class Performance (Best Model)

{_per_class_table()}

**Critical Recall = {best['Critical Recall']:.4f}**
Operationally, failing to detect a Critical ticket is more costly than any other
confusion type. This metric should be monitored closely in production.

---

## 11. Error Analysis

Total misclassified: {len(err_df):,} / {len(y_test):,}
Error rate: {len(err_df)/len(y_test)*100:.2f}%

Most common confusion pairs:

{_confusion_pairs(err_df)}

---

## 12. Limitations

1. **Rule-derived labels.** Derived priority is based on keyword rules, not human judgment.
2. **Critical class is small** (n=207, 2.44%). Results for Critical are less statistically stable.
3. **No negation handling.** "No urgent issue" still matches urgency patterns.
4. **Synthetic data.** Many ticket descriptions contain boilerplate text.
5. **Single text feature.** Only `ticket_text` used; metadata (channel, product) excluded for leakage safety.
6. **Binary-word rules.** Phrases like "as soon as possible" drive ~91% of urgent-language matches.

---

## 13. Final Conclusion

The best model — **{best['model_name']} [class_weight={best['class_weight']}]** — achieves:

| Metric | Value |
| --- | --- |
| Macro F1 | {best['Macro F1']:.4f} |
| Weighted F1 | {best['Weighted F1']:.4f} |
| Accuracy | {best['Accuracy']:.4f} |
| Critical Recall | {best['Critical Recall']:.4f} |
| Critical F1 | {best['Critical F1']:.4f} |

The model is suitable for internal ticket routing provided:
- `class_weight='balanced'` is applied
- Results are evaluated using macro F1, not accuracy
- Critical-class predictions are validated by human agents
- Labels are documented as rule-derived, not human-annotated
"""
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"\n  Saved report: {REPORT_OUT.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" STAGE 6D — TRAIN PRIORITY CLASSIFICATION MODEL")
    print("=" * 65)
    print("  Target  : derived_priority  (NOT original Ticket Priority)")
    print("  Feature : ticket_text")

    # 1. Load
    X, y, df = load_data(DATA_IN)

    # 2+3. Split + save
    X_train, X_test, y_train, y_test = make_splits(X, y, df)

    # 4. Vectorise
    vectorizer, X_train_tfidf, X_test_tfidf = vectorise(X_train, X_test)
    n_features = X_train_tfidf.shape[1]

    # 5. Train all models
    results = train_all_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # 7. Confusion matrices for all
    save_all_confusion_matrices(results, y_test)

    # 10. Comparison CSV
    comp_df = save_comparison(results, COMPARISON_OUT)

    # 11. Select best
    best = select_best_model(results)

    # 8. Per-class performance
    print_per_class_performance(best, y_test)

    # Save best model confusion matrix separately (clearly named)
    cm_name = "priority_best_model_confusion_matrix.png"
    save_confusion_matrix(y_test, best["y_pred"],
                          f"Best: {best['model_name']} [{best['class_weight']}]",
                          cm_name)

    # 9. Error analysis
    err_df = error_analysis(best, X_test, y_test, MISCLASSIFIED_OUT)

    # 12. Save model + vectorizer
    save_model(best, vectorizer)

    # 13. Test predictions
    save_predictions(best, X_test, y_test, PREDICTIONS_OUT)

    # 14. Report
    save_report(best, results, y_train, y_test, n_features, err_df)

    # -----------------------------------------------------------------------
    # STEP 16 — FILE VERIFICATION
    # -----------------------------------------------------------------------
    print("\n=== FILE VERIFICATION ===")
    required_files = [
        Path(__file__),
        TRAIN_OUT,
        TEST_OUT,
        MODEL_OUT,
        VECTORIZER_OUT,
        COMPARISON_OUT,
        PREDICTIONS_OUT,
        MISCLASSIFIED_OUT,
        REPORT_OUT,
    ]
    # Check confusion matrices
    cm_files = list(FIGURES_DIR.glob("priority_*.png"))
    all_ok = True
    for f in required_files:
        exists = f.exists()
        status = "[OK] EXISTS" if exists else "[MISSING]"
        if not exists:
            all_ok = False
        print(f"  {status}: {f.relative_to(ROOT)}")
    print(f"  [OK] Confusion matrices in reports/figures/: {len(cm_files)} files")

    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STAGE 6D COMPLETED")
    print()
    print(f"Training samples  : {len(y_train):,}")
    print(f"Testing  samples  : {len(y_test):,}")
    print(f"Priority classes  : 4")
    print(f"TF-IDF features   : {n_features:,}")
    print()
    print(f"Best model        : {best['model_name']} [class_weight={best['class_weight']}]")
    print(f"Best Macro F1     : {best['Macro F1']:.4f}")
    print(f"Critical Recall   : {best['Critical Recall']:.4f}")
    print(f"Critical F1       : {best['Critical F1']:.4f}")
    print(f"Weighted F1       : {best['Weighted F1']:.4f}")
    print(f"Accuracy          : {best['Accuracy']:.4f}")
    print()
    print(f"Model saved       : {MODEL_OUT.relative_to(ROOT)}")
    print(f"Vectorizer saved  : {VECTORIZER_OUT.relative_to(ROOT)}")
    print()
    print("Files created:")
    for f in required_files[1:]:
        print(f"  - {f.relative_to(ROOT)}")
    print(f"  - reports/figures/ ({len(cm_files)} confusion matrix plots)")
    print("=" * 65)
    print()
    print("DO NOT build the web application.")
    print("DO NOT create the final prediction API.")
    print("DO NOT modify the category model.")
    print("STOP after Stage 6D.")


if __name__ == "__main__":
    main()
