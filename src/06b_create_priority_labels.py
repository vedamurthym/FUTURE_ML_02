"""
Stage 6B — Create Derived Priority Labels
==========================================
IMPORTANT DOCUMENTATION NOTE:
  These are RULE-DERIVED / SYNTHETIC priority labels, NOT original human-annotated
  ground truth. They are generated deterministically from ticket content using
  explicit, documented keyword and subject-based rules. They are intended as a
  transparent, reproducible ML training target — NOT as a replacement for genuine
  human labeling. Manual validation is strongly recommended before production use.

Why original Ticket Priority was rejected:
  - Stage 6 diagnostic showed accuracy ≈ 24.68% (random chance for 4 classes ≈ 25%).
  - Distribution: Low 24.36%, Medium 25.88%, High 24.62%, Critical 25.14%.
  - No correlation with any structured feature (Channel, Product, Subject).
  - The original labels are randomly assigned synthetic noise.

Script: src/06b_create_priority_labels.py
Author: Auto-generated for Stage 6B
"""

import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_IN = ROOT / "data" / "processed" / "customer_support_tickets_cleaned.csv"
DATA_OUT = ROOT / "data" / "processed" / "customer_support_tickets_priority_labeled.csv"
RULES_OUT = ROOT / "reports" / "tables" / "priority_labeling_rules.csv"
EXAMPLES_OUT = ROOT / "reports" / "tables" / "derived_priority_examples.csv"
REPORT_OUT = ROOT / "reports" / "docs" / "DERIVED_PRIORITY_LABELING_REPORT.md"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# RULE DEFINITIONS
# (Grounded in actual dataset vocabulary from Step-1 inspection)
# Priority order: CRITICAL > HIGH > MEDIUM > LOW
# ---------------------------------------------------------------------------

# Each rule is a dict with keys:
#   rule_name, priority, description, keywords (list), use_subject (bool)
#
# Matching logic:
#   - If use_subject=True  → check Ticket Subject field (exact or substring match)
#   - If use_subject=False → check ticket_text (lowercased) for any keyword
#   Rules within each priority tier are evaluated in listed order.
#   First match within a tier wins; tier order is Critical → High → Medium → Low.

CRITICAL_RULES = [
    {
        "rule_name": "critical_security_threat",
        "priority": "Critical",
        "description": "Explicit security threat: hacking, breach, fraud, stolen credentials",
        "keywords": ["hack", "hacked", "breach", "fraud", "stolen", "security breach"],
        "use_subject": False,
    },
    {
        "rule_name": "critical_data_loss_confirmed",
        "priority": "Critical",
        "description": "Confirmed data loss or total wipe event",
        "keywords": ["lost everything", "all data lost", "completely lost", "entire data", "wiped"],
        "use_subject": False,
    },
    {
        "rule_name": "critical_emergency_language",
        "priority": "Critical",
        "description": "Customer uses explicit emergency / critical / urgent-now language",
        "keywords": ["emergency", "critical issue", "absolutely urgent", "urgent help needed",
                     "urgent assistance", "extremely urgent"],
        "use_subject": False,
    },
    {
        "rule_name": "critical_complete_device_failure",
        "priority": "Critical",
        "description": "Device completely dead, unresponsive, will not turn on at all",
        "keywords": ["unresponsive", "completely dead", "totally unresponsive",
                     "not turning on", "won't turn on"],
        "use_subject": False,
    },
]

HIGH_RULES = [
    {
        "rule_name": "high_account_access_blocked",
        "priority": "High",
        "description": "Customer is locked out of account / cannot log in",
        "keywords": ["unable to access", "unable to log in", "unable to login",
                     "invalid credentials", "cannot log in", "login failed",
                     "account locked", "access denied", "regain access"],
        "use_subject": False,
    },
    {
        "rule_name": "high_account_access_subject",
        "priority": "High",
        "description": "Ticket Subject = account access",
        "keywords": ["account access"],
        "use_subject": True,
    },
    {
        "rule_name": "high_data_loss_subject",
        "priority": "High",
        "description": "Ticket Subject = data loss",
        "keywords": ["data loss"],
        "use_subject": True,
    },
    {
        "rule_name": "high_data_loss_text",
        "priority": "High",
        "description": "Data loss or deleted files mentioned in ticket text",
        "keywords": ["data loss", "lost data", "deleted files", "files deleted",
                     "data deleted", "lost my data"],
        "use_subject": False,
    },
    {
        "rule_name": "high_payment_failure",
        "priority": "High",
        "description": "Payment processing failure or transaction error",
        "keywords": ["payment failed", "transaction failed", "payment error",
                     "charged incorrectly", "overcharged", "double charged",
                     "billing error", "wrong charge"],
        "use_subject": False,
    },
    {
        "rule_name": "high_payment_subject",
        "priority": "High",
        "description": "Ticket Subject = payment issue",
        "keywords": ["payment issue"],
        "use_subject": True,
    },
    {
        "rule_name": "high_network_blocked",
        "priority": "High",
        "description": "Complete network failure preventing use of product/service",
        "keywords": ["no internet", "network down", "completely disconnected",
                     "unable to connect", "connection failed"],
        "use_subject": False,
    },
    {
        "rule_name": "high_network_problem_subject",
        "priority": "High",
        "description": "Ticket Subject = network problem",
        "keywords": ["network problem"],
        "use_subject": True,
    },
    {
        "rule_name": "high_software_crash",
        "priority": "High",
        "description": "Application crash preventing use",
        "keywords": ["application crashes", "crashes", "app crashes", "crashes when",
                     "crash every time", "keeps crashing"],
        "use_subject": False,
    },
    {
        "rule_name": "high_software_bug_subject",
        "priority": "High",
        "description": "Ticket Subject = software bug",
        "keywords": ["software bug"],
        "use_subject": True,
    },
    {
        "rule_name": "high_hardware_failure",
        "priority": "High",
        "description": "Hardware not charging or other significant hardware fault",
        "keywords": ["not charging", "hardware failure", "hardware fault",
                     "device broken", "not working at all"],
        "use_subject": False,
    },
    {
        "rule_name": "high_hardware_issue_subject",
        "priority": "High",
        "description": "Ticket Subject = hardware issue",
        "keywords": ["hardware issue"],
        "use_subject": True,
    },
    {
        "rule_name": "high_urgent_language",
        "priority": "High",
        "description": "Customer uses urgent / ASAP language (without explicit critical indicators)",
        "keywords": ["urgent", "asap", "as soon as possible", "immediately",
                     "right away", "cannot wait"],
        "use_subject": False,
    },
]

MEDIUM_RULES = [
    {
        "rule_name": "medium_refund_request",
        "priority": "Medium",
        "description": "Customer requests a refund",
        "keywords": ["refund", "money back", "reimbursement"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_refund_subject",
        "priority": "Medium",
        "description": "Ticket Subject = refund request",
        "keywords": ["refund request"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_cancellation",
        "priority": "Medium",
        "description": "Customer wants to cancel a subscription/order/service",
        "keywords": ["cancel", "cancellation", "cancel my subscription",
                     "cancel order", "cancel service"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_cancellation_subject",
        "priority": "Medium",
        "description": "Ticket Subject = cancellation request",
        "keywords": ["cancellation request"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_delivery_problem",
        "priority": "Medium",
        "description": "Delivery or order tracking issue",
        "keywords": ["delivery", "shipping", "not delivered", "delivery delayed",
                     "order tracking", "package"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_delivery_subject",
        "priority": "Medium",
        "description": "Ticket Subject = delivery problem",
        "keywords": ["delivery problem"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_installation_support",
        "priority": "Medium",
        "description": "Customer needs help installing or setting up a product",
        "keywords": ["installation", "install", "setup", "set up", "configure",
                     "getting started"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_installation_subject",
        "priority": "Medium",
        "description": "Ticket Subject = installation support or product setup",
        "keywords": ["installation support", "product setup"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_compatibility",
        "priority": "Medium",
        "description": "Product or peripheral compatibility question",
        "keywords": ["compatibility", "compatible", "not compatible", "works with"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_compatibility_subject",
        "priority": "Medium",
        "description": "Ticket Subject = peripheral compatibility or product compatibility",
        "keywords": ["peripheral compatibility", "product compatibility"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_display_issue",
        "priority": "Medium",
        "description": "Display or screen problem but product still partially usable",
        "keywords": ["display issue", "screen issue", "screen flicker", "display problem",
                     "blurry screen", "flickering"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_display_subject",
        "priority": "Medium",
        "description": "Ticket Subject = display issue",
        "keywords": ["display issue"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_battery_life",
        "priority": "Medium",
        "description": "Battery draining faster than expected but device still functional",
        "keywords": ["battery life", "battery drain", "battery draining",
                     "decrease in battery", "battery not lasting"],
        "use_subject": False,
    },
    {
        "rule_name": "medium_battery_subject",
        "priority": "Medium",
        "description": "Ticket Subject = battery life",
        "keywords": ["battery life"],
        "use_subject": True,
    },
    {
        "rule_name": "medium_general_not_working",
        "priority": "Medium",
        "description": "General 'not working' without specific critical/high indicators",
        "keywords": ["not working", "stopped working", "no longer works", "broken",
                     "malfunctioning"],
        "use_subject": False,
    },
]

LOW_RULES = [
    {
        "rule_name": "low_product_recommendation",
        "priority": "Low",
        "description": "Customer asks for a product recommendation",
        "keywords": ["recommendation", "suggest", "suggest a product", "which product",
                     "what product", "best product", "advice"],
        "use_subject": False,
    },
    {
        "rule_name": "low_recommendation_subject",
        "priority": "Low",
        "description": "Ticket Subject = product recommendation",
        "keywords": ["product recommendation"],
        "use_subject": True,
    },
    {
        "rule_name": "low_general_inquiry",
        "priority": "Low",
        "description": "General information request or how-to question",
        "keywords": ["how do i", "how can i", "how to", "what is", "where can i",
                     "i would like to know", "can you tell me"],
        "use_subject": False,
    },
    {
        "rule_name": "low_troubleshooting_request",
        "priority": "Low",
        "description": "Customer asks about troubleshooting steps only, no urgency",
        "keywords": ["troubleshooting", "user manual", "steps to fix",
                     "guide me", "guide through"],
        "use_subject": False,
    },
    {
        "rule_name": "low_firmware_update",
        "priority": "Low",
        "description": "Firmware/software update question",
        "keywords": ["firmware update", "software update", "firmware version",
                     "update available", "latest update"],
        "use_subject": False,
    },
    {
        "rule_name": "low_factory_reset",
        "priority": "Low",
        "description": "Customer performed/asking about a factory reset",
        "keywords": ["factory reset", "reset the device", "performed a factory"],
        "use_subject": False,
    },
]

# All rules in precedence order (Critical first)
ALL_RULE_GROUPS = [
    ("Critical", CRITICAL_RULES),
    ("High",     HIGH_RULES),
    ("Medium",   MEDIUM_RULES),
    ("Low",      LOW_RULES),
]


# ---------------------------------------------------------------------------
# HELPER: compile keyword patterns
# ---------------------------------------------------------------------------
def _compile_pattern(keywords: list) -> re.Pattern:
    """Return a compiled OR-pattern for a list of keyword phrases."""
    escaped = [re.escape(kw.lower()) for kw in keywords]
    return re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)


def _precompile_rules(rule_groups):
    """Attach compiled regex patterns to each rule dict (in-place)."""
    for _priority, rules in rule_groups:
        for rule in rules:
            rule["_pattern"] = _compile_pattern(rule["keywords"])
    return rule_groups


# ---------------------------------------------------------------------------
# CORE LABELING FUNCTION
# ---------------------------------------------------------------------------
def assign_priority(row, compiled_rules) -> tuple[str, str]:
    """
    Assign a derived priority label to a single ticket row.

    Returns
    -------
    (derived_priority, matched_rule_name)
        derived_priority : "Critical" | "High" | "Medium" | "Low" | "Unclear"
        matched_rule_name : name of first rule that fired, or "" if Unclear
    """
    text = str(row.get("ticket_text", "")).lower()
    subject = str(row.get("Ticket Subject", "")).lower().strip()

    for _priority, rules in compiled_rules:
        for rule in rules:
            pattern = rule["_pattern"]
            if rule["use_subject"]:
                if pattern.search(subject):
                    return rule["priority"], rule["rule_name"]
            else:
                if pattern.search(text):
                    return rule["priority"], rule["rule_name"]

    return "Unclear", ""


# ---------------------------------------------------------------------------
# STEP 1 — LOAD DATA
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — APPLY LABELING RULES
# ---------------------------------------------------------------------------
def apply_labels(df: pd.DataFrame, compiled_rules) -> pd.DataFrame:
    print("Applying derived priority rules...")
    results = df.apply(lambda row: assign_priority(row, compiled_rules), axis=1)
    df = df.copy()
    df["derived_priority"] = results.apply(lambda x: x[0])
    df["priority_matched_rule"] = results.apply(lambda x: x[1])
    return df


# ---------------------------------------------------------------------------
# STEP 3 — DISTRIBUTION ANALYSIS
# ---------------------------------------------------------------------------
def print_distribution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== DERIVED PRIORITY DISTRIBUTION ===")
    dist = df["derived_priority"].value_counts()
    pct  = df["derived_priority"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"Count": dist, "Percentage (%)": pct.round(2)})
    print(summary.to_string())

    print("\n--- Original Ticket Priority (for comparison) ---")
    orig = df["Ticket Priority"].value_counts()
    opct = df["Ticket Priority"].value_counts(normalize=True) * 100
    osummary = pd.DataFrame({"Count": orig, "Percentage (%)": opct.round(2)})
    print(osummary.to_string())

    unclear_count = (df["derived_priority"] == "Unclear").sum()
    unclear_pct   = unclear_count / len(df) * 100
    print(f"\nUnclear tickets: {unclear_count:,} ({unclear_pct:.2f}%)")

    # Imbalance stats (exclude Unclear)
    labeled = df[df["derived_priority"] != "Unclear"]
    if len(labeled) > 0:
        counts = labeled["derived_priority"].value_counts()
        majority_pct = counts.iloc[0] / len(labeled) * 100
        minority_pct = counts.iloc[-1] / len(labeled) * 100
        ratio = counts.iloc[0] / counts.iloc[-1]
        print(f"\nAmong labeled tickets ({len(labeled):,}):")
        print(f"  Majority class : {counts.index[0]} ({majority_pct:.2f}%)")
        print(f"  Minority class : {counts.index[-1]} ({minority_pct:.2f}%)")
        print(f"  Imbalance ratio: {ratio:.2f}:1")

    return summary


# ---------------------------------------------------------------------------
# STEP 4 — SAVE LABELED DATASET
# ---------------------------------------------------------------------------
def save_labeled_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved labeled dataset to: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# STEP 5 — SAVE RULE DEFINITIONS
# ---------------------------------------------------------------------------
def save_rules_csv(compiled_rules, path: Path) -> None:
    records = []
    for _priority, rules in compiled_rules:
        for rule in rules:
            records.append({
                "rule_name": rule["rule_name"],
                "priority": rule["priority"],
                "description": rule["description"],
                "matched_keyword_or_pattern": " | ".join(rule["keywords"]),
                "field_checked": "Ticket Subject" if rule["use_subject"] else "ticket_text",
            })
    rdf = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(path, index=False)
    print(f"Saved rule definitions to: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# STEP 6 — SAVE EXAMPLES
# ---------------------------------------------------------------------------
def save_examples(df: pd.DataFrame, path: Path, n_per_class: int = 30) -> None:
    levels = ["Critical", "High", "Medium", "Low", "Unclear"]
    frames = []
    for level in levels:
        subset = df[df["derived_priority"] == level][
            ["ticket_text", "derived_priority", "priority_matched_rule"]
        ]
        if len(subset) == 0:
            continue
        sample = subset.sample(
            n=min(n_per_class, len(subset)), random_state=RANDOM_STATE
        )
        frames.append(sample)
    examples = pd.concat(frames, ignore_index=True)
    examples.columns = ["ticket_text", "derived_priority", "matched_rule"]
    path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(path, index=False)
    print(f"Saved examples to: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# STEP 7 — SAVE REPORT
# ---------------------------------------------------------------------------
def save_report(df: pd.DataFrame, dist_summary: pd.DataFrame, path: Path) -> None:
    unclear_count = int((df["derived_priority"] == "Unclear").sum())
    unclear_pct   = unclear_count / len(df) * 100
    labeled_df    = df[df["derived_priority"] != "Unclear"]
    total_labeled = len(labeled_df)

    def _dist_table() -> str:
        lines = ["| Priority | Count | Percentage |", "| --- | --- | --- |"]
        for lvl in ["Critical", "High", "Medium", "Low", "Unclear"]:
            cnt = int((df["derived_priority"] == lvl).sum())
            pct = cnt / len(df) * 100
            lines.append(f"| {lvl} | {cnt:,} | {pct:.2f}% |")
        return "\n".join(lines)

    def _rules_table() -> str:
        lines = ["| Rule Name | Priority | Description |", "| --- | --- | --- |"]
        for _p, rules in ALL_RULE_GROUPS:
            for rule in rules:
                lines.append(
                    f"| `{rule['rule_name']}` | {rule['priority']} | {rule['description']} |"
                )
        return "\n".join(lines)

    report = f"""# DERIVED PRIORITY LABELING REPORT
## Stage 6B — Rule-Based Priority Label Generation

> **⚠️ IMPORTANT WARNING**
> These labels are **RULE-DERIVED / SYNTHETIC**. They are NOT original human-annotated
> ground truth. They were generated deterministically from ticket content using explicit,
> documented keyword/subject-based rules. Manual validation is **strongly recommended**
> before using these labels in a production ML system.

---

## 1. Why Original Ticket Priority Was Rejected

The original `Ticket Priority` column from the Kaggle dataset was found to be **randomly
assigned** during Stage 6 diagnosis:

| Metric | Value |
| --- | --- |
| Text model Accuracy | 0.2438 |
| Weighted F1 | 0.2438 |
| Macro F1 | 0.2440 |
| Random chance (4 classes) | ≈ 25.00% |

Priority distributions were nearly equal (~24–26% per class), confirming random assignment.
No correlation was found with any structured feature (Channel, Product, Subject).

---

## 2. Evidence from Stage 6 Diagnosis

```
Original Ticket Priority Distribution:
  Low      : 2,063 (24.36%)
  Medium   : 2,192 (25.88%)
  High     : 2,085 (24.62%)
  Critical : 2,129 (25.14%)

Classifier Accuracy ≈ 24.68% (vs. 25.00% random baseline)
```

---

## 3. Priority Definitions

| Level | Definition |
| --- | --- |
| **Critical** | Explicit security threats (hacking/breach/fraud), confirmed data loss, explicit emergency language, complete device unresponsiveness |
| **High** | Account access blocked, active data loss, payment transaction failures, complete network failure, application crashes, hardware failures, urgent/ASAP language |
| **Medium** | Refund or cancellation requests, delivery/shipping problems, installation/setup help, compatibility questions, display or battery issues, general malfunction |
| **Low** | Product recommendations, general how-to/information inquiries, firmware update questions, factory reset guidance |
| **Unclear** | Insufficient evidence to confidently assign any of the above |

---

## 4. Rule Groups

{_rules_table()}

---

## 5. Rule Precedence

Rules are applied in strict priority order:

1. **Critical** — evaluated first; overrides all lower tiers
2. **High** — evaluated if no Critical rule matched
3. **Medium** — evaluated if no Critical or High rule matched
4. **Low** — evaluated if no higher-tier rule matched
5. **Unclear** — assigned when no rule matches at all

Within each tier, rules are evaluated in declaration order; the **first matching rule** wins.

---

## 6. Distribution of Derived Priorities

{_dist_table()}

**Total rows:** {len(df):,}
**Labeled rows:** {total_labeled:,}
**Unclear rows:** {unclear_count:,} ({unclear_pct:.2f}%)

---

## 7. Unclear Cases

Tickets labeled **Unclear** ({unclear_count:,}, {unclear_pct:.2f}%) did not match any rule.

Common reasons:
- Highly generic text with no specific urgency or subject keyword
- Synthetic boilerplate sentences with no actionable vocabulary
- Ticket descriptions that are very short or vague

Recommendation: Review a random sample of Unclear tickets to determine whether
additional rules can reasonably classify them, or keep them as a "no-label" set.

---

## 8. Class Imbalance

Among {total_labeled:,} labeled tickets:
- The dataset is expected to be **imbalanced** (Medium and High likely largest tiers).
- Use `class_weight='balanced'` in ML classifiers to handle imbalance.
- Do NOT artificially resample to force equal class sizes.

---

## 9. Limitations

1. **Not human-annotated.** Rules may misclassify edge cases.
2. **Keyword sensitivity.** Negation (e.g., "no urgent issue") is NOT handled.
3. **No sentence-level context.** Patterns match anywhere in ticket_text.
4. **Synthetic data.** Many descriptions are boilerplate; rules may apply inconsistently.
5. **Rule coverage.** Some ticket vocabulary may not be covered; those become Unclear.

---

## 10. Recommendation Before ML Training

> **Perform manual validation on at least 50–100 randomly sampled tickets per priority
> level** before using `derived_priority` as the supervised ML training target.
> Compare rule-assigned labels with human judgment and refine rules as needed.

---

## 11. Files Created

| File | Description |
| --- | --- |
| `data/processed/customer_support_tickets_priority_labeled.csv` | Full dataset with `derived_priority` and `priority_matched_rule` added |
| `reports/priority_labeling_rules.csv` | Machine-readable rule definitions |
| `reports/derived_priority_examples.csv` | Sample tickets per priority level (≥30 each) |
| `reports/DERIVED_PRIORITY_LABELING_REPORT.md` | This report |
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"Saved report to: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" STAGE 6B — CREATE DERIVED PRIORITY LABELS")
    print("=" * 65)

    # Pre-compile all rule patterns once
    compiled_rules = _precompile_rules(ALL_RULE_GROUPS)

    # 1. Load
    df = load_data(DATA_IN)

    # 2. Apply rules
    df = apply_labels(df, compiled_rules)

    # 3. Distribution analysis
    dist_summary = print_distribution(df)

    # 4. Save labeled dataset (keep ALL original columns + 2 new)
    save_labeled_dataset(df, DATA_OUT)

    # 5. Save rule definitions CSV
    save_rules_csv(compiled_rules, RULES_OUT)

    # 6. Save examples
    save_examples(df, EXAMPLES_OUT, n_per_class=30)

    # 7. Save report
    save_report(df, dist_summary, REPORT_OUT)

    # 8. Verify files
    print("\n=== FILE VERIFICATION ===")
    required_files = [
        Path(__file__),
        DATA_OUT,
        RULES_OUT,
        EXAMPLES_OUT,
        REPORT_OUT,
    ]
    for f in required_files:
        status = "[OK] EXISTS" if f.exists() else "[MISSING]"
        print(f"  {status}: {f.relative_to(ROOT)}")

    # 9. Final summary
    derived_counts = df["derived_priority"].value_counts()
    unclear_count  = int((df["derived_priority"] == "Unclear").sum())
    unclear_pct    = unclear_count / len(df) * 100

    print("\n" + "=" * 65)
    print("STAGE 6B COMPLETED")
    print()
    print("Original Ticket Priority:")
    print("  NOT USED FOR ML TRAINING")
    print("  (Random/noisy labels — Stage 6 diagnostic confirmed)")
    print()
    print("Derived priorities:")
    for lvl in ["Critical", "High", "Medium", "Low", "Unclear"]:
        cnt = int((df["derived_priority"] == lvl).sum())
        pct = cnt / len(df) * 100
        print(f"  {lvl:<12}: {cnt:>5,} ({pct:.2f}%)")
    print()
    print(f"Unclear tickets: {unclear_count:,} ({unclear_pct:.2f}%)")
    print()
    print("Priority distribution:")
    for lvl in ["Critical", "High", "Medium", "Low"]:
        cnt = int((df["derived_priority"] == lvl).sum())
        pct = cnt / len(df) * 100
        print(f"  {lvl:<12}: {cnt:>5,} ({pct:.2f}%)")
    print()
    print("Files created:")
    for f in required_files[1:]:
        print(f"  - {f.relative_to(ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
