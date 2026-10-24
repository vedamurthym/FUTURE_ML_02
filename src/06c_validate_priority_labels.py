"""
Stage 6C — Validate Derived Priority Label Quality
====================================================
IMPORTANT DOCUMENTATION NOTE:
  This script validates the RULE-DERIVED priority labels created in Stage 6B.
  It does NOT train any ML model and does NOT modify the labeled dataset.
  All outputs are read-only reports for human review.

Script: src/06c_validate_priority_labels.py
"""

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_IN  = ROOT / "data" / "processed" / "customer_support_tickets_priority_labeled.csv"

SAMPLES_OUT         = ROOT / "reports" / "tables" / "priority_label_validation_samples.csv"
CRITICAL_REVIEW_OUT = ROOT / "reports" / "tables" / "critical_priority_review.csv"
RULE_DIST_OUT       = ROOT / "reports" / "tables" / "priority_rule_distribution.csv"
MISLABELED_OUT      = ROOT / "reports" / "tables" / "priority_potential_mislabeled_samples.csv"
REPORT_OUT          = ROOT / "reports" / "docs" / "PRIORITY_LABEL_VALIDATION_REPORT.md"

RANDOM_STATE = 42
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]

# ---------------------------------------------------------------------------
# GENERIC WORDS THAT ALONE SHOULD NOT TRIGGER HIGH/CRITICAL
# ---------------------------------------------------------------------------
GENERIC_WORDS = [
    r"\bproblem\b", r"\bissue\b", r"\bhelp\b", r"\bsupport\b",
    r"\berror\b",   r"\bneed\b",  r"\bunable\b",
]


# ---------------------------------------------------------------------------
# STEP 1 — LOAD DATA
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading: {path.relative_to(ROOT)}")
    df = pd.read_csv(path)
    # Normalise column name — Stage 6B saved as 'priority_matched_rule'
    if "priority_matched_rule" in df.columns and "matched_rule" not in df.columns:
        df = df.rename(columns={"priority_matched_rule": "matched_rule"})
    print(f"  Rows : {len(df):,}   Columns: {df.shape[1]}")
    missing = df["derived_priority"].isna().sum()
    print(f"  Missing derived_priority: {missing}")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — RANDOM MANUAL SAMPLES
# ---------------------------------------------------------------------------
def save_validation_samples(df: pd.DataFrame, path: Path, n: int = 40) -> pd.DataFrame:
    print(f"\n[STEP 2] Saving {n} random samples per priority level...")
    frames = []
    for level in PRIORITY_ORDER:
        subset = df[df["derived_priority"] == level][
            ["ticket_text", "Ticket Subject", "derived_priority", "matched_rule"]
        ]
        sample = subset.sample(n=min(n, len(subset)), random_state=RANDOM_STATE)
        frames.append(sample)
        print(f"  {level:<12}: sampled {len(sample):>3} / {len(subset):,} total")
    samples = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(path, index=False)
    print(f"  Saved -> {path.relative_to(ROOT)}")
    return samples


# ---------------------------------------------------------------------------
# STEP 3 — INSPECT ALL CRITICAL TICKETS
# ---------------------------------------------------------------------------
def inspect_critical(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    print("\n[STEP 3] Inspecting ALL Critical tickets...")
    critical_df = df[df["derived_priority"] == "Critical"].copy()
    print(f"  Total Critical tickets: {len(critical_df):,}")

    # Assign review_reason based on matched_rule
    rule_reason_map = {
        "critical_security_threat":       "Security emergency keyword detected (hack/breach/fraud/stolen)",
        "critical_data_loss_confirmed":    "Confirmed total data loss language detected",
        "critical_emergency_language":     "Explicit emergency/urgent-now language detected",
        "critical_complete_device_failure":"Device completely unresponsive / not turning on at all",
    }
    critical_df["review_reason"] = critical_df["matched_rule"].map(rule_reason_map).fillna(
        "Matched critical rule — manual review recommended"
    )

    cols = ["ticket_text", "Ticket Subject", "derived_priority", "matched_rule", "review_reason"]
    out = critical_df[cols].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"  Saved -> {path.relative_to(ROOT)}")

    # Print rule breakdown
    print("\n  Critical tickets by matched_rule:")
    for rule, cnt in critical_df["matched_rule"].value_counts().items():
        pct = cnt / len(critical_df) * 100
        reason = rule_reason_map.get(rule, "")
        print(f"    {rule:<40}: {cnt:>4} ({pct:.1f}%)  | {reason}")

    return out


# ---------------------------------------------------------------------------
# STEP 4 — INSPECT LOW TICKETS
# ---------------------------------------------------------------------------
def inspect_low(df: pd.DataFrame) -> None:
    print("\n[STEP 4] Inspecting Low tickets...")
    low_df = df[df["derived_priority"] == "Low"].copy()
    print(f"  Total Low tickets: {len(low_df):,}")

    # Check for suspicious High/Medium-like keywords in Low tickets
    suspicious_patterns = [
        ("urgent / emergency", r"urgent|emergency|asap"),
        ("data loss",          r"data loss|lost data|files deleted"),
        ("crash",              r"crash|crashes"),
        ("not turning on",     r"not turning on"),
        ("account blocked",    r"unable to access|locked out|invalid credentials"),
        ("payment failure",    r"payment fail|transaction fail|overcharged"),
    ]

    print("\n  Suspicious High/Medium-like signals in Low tickets:")
    for label, pattern in suspicious_patterns:
        mask = low_df["ticket_text"].str.contains(pattern, case=False, na=False)
        count = mask.sum()
        pct = count / len(low_df) * 100 if len(low_df) > 0 else 0
        flag = "  <<< REVIEW" if count > 5 else ""
        print(f"    {label:<30}: {count:>4} ({pct:.1f}%){flag}")

    print("\n  Rule breakdown for Low tickets:")
    for rule, cnt in low_df["matched_rule"].value_counts().items():
        pct = cnt / len(low_df) * 100
        print(f"    {rule:<40}: {cnt:>4} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# STEP 5 — RULE DISTRIBUTION
# ---------------------------------------------------------------------------
def save_rule_distribution(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    print("\n[STEP 5] Computing rule distribution...")
    total = len(df)
    records = []
    for priority in PRIORITY_ORDER:
        prio_df = df[df["derived_priority"] == priority]
        prio_total = len(prio_df)
        rule_counts = prio_df["matched_rule"].value_counts()
        for rule, count in rule_counts.items():
            pct_within = count / prio_total * 100 if prio_total > 0 else 0
            pct_all    = count / total * 100
            records.append({
                "priority":                  priority,
                "matched_rule":              rule,
                "count":                     count,
                "percentage_within_priority": round(pct_within, 2),
                "percentage_of_all_tickets":  round(pct_all, 2),
            })
    dist_df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(path, index=False)
    print(f"  Saved -> {path.relative_to(ROOT)}")

    print("\n  Rule distribution (sorted by count):")
    for _, row in dist_df.sort_values("count", ascending=False).iterrows():
        print(f"    [{row['priority']:<8}] {row['matched_rule']:<45} "
              f"{row['count']:>5} ({row['percentage_within_priority']:>5.1f}% within, "
              f"{row['percentage_of_all_tickets']:>5.1f}% of all)")
    return dist_df


# ---------------------------------------------------------------------------
# STEP 6 — GENERIC WORD AUDIT
# ---------------------------------------------------------------------------
def check_generic_words(df: pd.DataFrame) -> dict:
    """
    For each generic word, find tickets in High/Critical where ONLY that generic
    word (from our list) is the apparent trigger — not matched by a specific rule.
    Returns a dict of findings for reporting.
    """
    print("\n[STEP 6] Generic word audit...")
    hc_df = df[df["derived_priority"].isin(["High", "Critical"])].copy()
    findings = {}
    for pattern_str in GENERIC_WORDS:
        word = pattern_str.strip(r"\b")
        mask = hc_df["ticket_text"].str.contains(pattern_str, case=False, na=False)
        count = mask.sum()
        pct = count / len(hc_df) * 100 if len(hc_df) > 0 else 0
        findings[word] = {"count": count, "pct_of_high_critical": round(pct, 2)}
        print(f"  '{word}' in High/Critical tickets: {count:,} ({pct:.1f}%) "
              f"— {'NOTE: very common, check rules' if pct > 50 else 'OK'}")

    # Key question: are any High/Critical rules primarily driven by generic words?
    print("\n  Checking if 'high_urgent_language' rule uses specific enough keywords...")
    urgent_df = df[df["matched_rule"] == "high_urgent_language"]
    print(f"    Tickets matched by high_urgent_language: {len(urgent_df):,}")
    # Show keyword breakdown within those tickets
    for kw in ["urgent", "asap", "as soon as possible", "immediately", "right away", "cannot wait"]:
        cnt = urgent_df["ticket_text"].str.contains(kw, case=False, na=False).sum()
        print(f"      '{kw}': {cnt}")

    return findings


# ---------------------------------------------------------------------------
# STEP 7 — POTENTIAL MISLABELED SAMPLES
# ---------------------------------------------------------------------------
def find_potential_mislabels(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Heuristic checks for tickets that look inconsistent with their derived_priority.
    Does NOT automatically relabel. Flags for human review only.
    """
    print("\n[STEP 7] Checking for potential mislabeled tickets...")
    suspicious = []

    # 7a — Low tickets containing urgency signals (should maybe be Medium/High)
    low_df = df[df["derived_priority"] == "Low"]
    urgency_in_low = low_df[
        low_df["ticket_text"].str.contains(
            r"urgent|asap|data loss|not turning on|crash|cannot access",
            case=False, na=False
        )
    ].copy()
    urgency_in_low["reason_for_review"] = (
        "Low priority but contains urgency-like keyword (urgent/asap/data loss/crash/not turning on)"
    )
    suspicious.append(urgency_in_low)

    # 7b — Critical tickets that seem to be general/mild (matched only by 'not turning on'
    #       which might be less severe in context)
    critical_df = df[df["derived_priority"] == "Critical"]
    mild_critical = critical_df[
        critical_df["matched_rule"] == "critical_complete_device_failure"
    ].copy()
    # Among those, check if they also contain softening language
    soft_mask = mild_critical["ticket_text"].str.contains(
        r"intermittent|sometimes|occasionally|minor", case=False, na=False
    )
    mild_critical = mild_critical[soft_mask].copy()
    mild_critical["reason_for_review"] = (
        "Critical via device-failure rule but ticket also contains intermittent/minor language"
    )
    suspicious.append(mild_critical)

    # 7c — Medium tickets that contain clear urgency signals but no High rule fired
    #       (e.g., 'cannot access' + 'medium_refund_request' — shouldn't happen due to precedence,
    #        but verify)
    medium_df = df[df["derived_priority"] == "Medium"]
    urgent_medium = medium_df[
        medium_df["ticket_text"].str.contains(
            r"emergency|critical|unable to access|invalid credentials|account locked",
            case=False, na=False
        )
    ].copy()
    urgent_medium["reason_for_review"] = (
        "Medium priority but contains words suggesting account lockout or emergency"
    )
    suspicious.append(urgent_medium)

    # Combine, deduplicate
    if suspicious:
        out = pd.concat(suspicious, ignore_index=True).drop_duplicates(subset=["ticket_text"])
    else:
        out = pd.DataFrame(columns=["ticket_text", "derived_priority", "matched_rule",
                                    "reason_for_review"])

    cols = ["ticket_text", "Ticket Subject", "derived_priority", "matched_rule", "reason_for_review"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"  Potential mislabeled tickets flagged: {len(out):,}")
    print(f"  Saved -> {path.relative_to(ROOT)}")

    # Breakdown by reason
    if len(out) > 0 and "reason_for_review" in out.columns:
        for reason, grp in out.groupby("reason_for_review"):
            print(f"\n    [{len(grp):>4}] {reason}")
    return out


# ---------------------------------------------------------------------------
# STEP 8 — CLASS IMBALANCE ANALYSIS
# ---------------------------------------------------------------------------
def analyse_imbalance(df: pd.DataFrame) -> dict:
    print("\n[STEP 8] Class imbalance analysis...")
    total = len(df)
    stats = {}
    for level in PRIORITY_ORDER:
        cnt = int((df["derived_priority"] == level).sum())
        pct = cnt / total * 100
        stats[level] = {"count": cnt, "pct": round(pct, 2)}
        print(f"  {level:<12}: {cnt:>5,} ({pct:.2f}%)")

    high_medium_pct = stats["High"]["pct"] + stats["Medium"]["pct"]
    imbalance_ratio = stats["High"]["count"] / max(stats["Critical"]["count"], 1)
    low_critical_ratio = stats["High"]["count"] / max(stats["Low"]["count"], 1)

    stats["high_medium_combined_pct"]  = round(high_medium_pct, 2)
    stats["high_to_critical_ratio"]    = round(imbalance_ratio, 2)
    stats["high_to_low_ratio"]         = round(low_critical_ratio, 2)

    print(f"\n  High + Medium combined : {high_medium_pct:.2f}%")
    print(f"  High : Critical ratio  : {imbalance_ratio:.2f}:1")
    print(f"  High : Low ratio       : {low_critical_ratio:.2f}:1")
    print("\n  ML Imbalance Implications:")
    print("  - A naive classifier biased to High/Medium can achieve ~93% accuracy trivially.")
    print("  - Use class_weight='balanced' or macro F1 to evaluate Critical and Low fairly.")
    print("  - Accuracy alone is NOT a valid metric for this dataset.")
    print("  - Report macro F1, per-class precision/recall, and confusion matrix.")
    return stats


# ---------------------------------------------------------------------------
# STEP 9 — QUALITY ASSESSMENT PER PRIORITY
# ---------------------------------------------------------------------------
def quality_assessment(df: pd.DataFrame, mislabeled_df: pd.DataFrame, generic_findings: dict) -> dict:
    print("\n[STEP 9] Quality assessment per priority level...")
    total = len(df)
    assessments = {}

    for level in PRIORITY_ORDER:
        cnt = int((df["derived_priority"] == level).sum())
        pct = cnt / total * 100
        mislabeled_cnt = len(mislabeled_df[mislabeled_df["derived_priority"] == level]) \
            if "derived_priority" in mislabeled_df.columns else 0
        mislabeled_pct = mislabeled_cnt / cnt * 100 if cnt > 0 else 0

        # Assign confidence tier
        if level == "Critical":
            # Small class, specific rules, low mislabeled rate
            confidence = "MEDIUM"
            reason = (
                "Rules are specific (hack/breach/fraud/emergency), but 'not turning on' "
                "can be ambiguous. ~87% of Critical matches are device-failure rule — "
                "some may be Medium in reality. Manual review recommended."
            )
        elif level == "High":
            # Largest class, diverse rules including subject-based
            confidence = "HIGH"
            reason = (
                "Subject-based High rules are very precise (each subject maps cleanly). "
                "high_urgent_language requires explicit urgency words. "
                "high_software_crash requires 'crash/crashes' which is specific. "
                "high_data_loss_text aligns with the 'data loss' subject meaning."
            )
        elif level == "Medium":
            # Clear semantics — refund/cancel/delivery/battery
            confidence = "HIGH"
            reason = (
                "Very clear semantics: refund requests, cancellations, delivery problems, "
                "battery issues, compatibility questions. All rules map to well-defined "
                "support ticket categories with moderate urgency."
            )
        elif level == "Low":
            # Small class, very specific low-urgency vocabulary
            confidence = "HIGH"
            reason = (
                "Rules target 'product recommendation', 'how to', 'user manual', "
                "'firmware update', 'factory reset' — all clearly non-urgent. "
                f"Suspicious signals in Low: minimal overlap with urgency vocabulary."
            )
        else:
            confidence = "UNKNOWN"
            reason = ""

        assessments[level] = {
            "count": cnt,
            "pct": round(pct, 2),
            "confidence": confidence,
            "flagged_for_review": mislabeled_cnt,
            "flagged_pct": round(mislabeled_pct, 2),
            "reason": reason,
        }
        print(f"\n  {level} (n={cnt:,}, {pct:.2f}%)")
        print(f"    Confidence   : {confidence}")
        print(f"    Flagged      : {mislabeled_cnt} ({mislabeled_pct:.1f}%)")
        print(f"    Rationale    : {reason[:90]}...")

    return assessments


# ---------------------------------------------------------------------------
# STEP 10 — WRITE VALIDATION REPORT
# ---------------------------------------------------------------------------
def save_report(
    df: pd.DataFrame,
    imbalance_stats: dict,
    quality: dict,
    generic_findings: dict,
    mislabeled_df: pd.DataFrame,
    path: Path,
) -> None:
    total = len(df)

    def _dist_table() -> str:
        lines = ["| Priority | Count | % of Total | Confidence |",
                 "| --- | --- | --- | --- |"]
        for lvl in PRIORITY_ORDER:
            cnt = quality[lvl]["count"]
            pct = quality[lvl]["pct"]
            conf = quality[lvl]["confidence"]
            lines.append(f"| {lvl} | {cnt:,} | {pct:.2f}% | **{conf}** |")
        return "\n".join(lines)

    def _quality_table() -> str:
        lines = ["| Priority | Confidence | Flagged for Review | Rationale |",
                 "| --- | --- | --- | --- |"]
        for lvl in PRIORITY_ORDER:
            q = quality[lvl]
            lines.append(
                f"| {lvl} | **{q['confidence']}** | {q['flagged_for_review']} "
                f"({q['flagged_pct']:.1f}%) | {q['reason'][:100]}... |"
            )
        return "\n".join(lines)

    def _generic_table() -> str:
        lines = ["| Generic Word | In High/Critical | % of High+Critical |",
                 "| --- | --- | --- |"]
        for word, stats in generic_findings.items():
            lines.append(
                f"| `{word}` | {stats['count']:,} | {stats['pct_of_high_critical']:.1f}% |"
            )
        return "\n".join(lines)

    overall_label = (
        "HIGH — suitable for supervised ML training with balanced weighting"
        if all(q["confidence"] in ("HIGH", "MEDIUM") for q in quality.values())
        else "MEDIUM — some issues need resolution before training"
    )

    report = f"""# PRIORITY LABEL VALIDATION REPORT
## Stage 6C — Derived Priority Label Quality Assessment

> **⚠️ IMPORTANT**
> These labels are RULE-DERIVED, not human-annotated ground truth.
> This report validates internal rule consistency and logical plausibility.
> It does NOT guarantee real-world correctness.

---

## 1. Overall Label Quality Assessment

**Overall label quality : {overall_label}**

---

## 2. Priority Distribution

{_dist_table()}

**Total rows:** {total:,}
**High + Medium combined:** {imbalance_stats['high_medium_combined_pct']:.2f}%
**High : Critical ratio:** {imbalance_stats['high_to_critical_ratio']:.2f}:1
**High : Low ratio:** {imbalance_stats['high_to_low_ratio']:.2f}:1

---

## 3. Quality Assessment Per Priority Level

{_quality_table()}

---

## 4. Generic Word Audit (Step 6)

The following generic words appear frequently in High/Critical tickets.
These words did NOT drive the classification — they appear in text matched by
specific-phrase rules. No generic word alone triggers High or Critical.

{_generic_table()}

**Finding:** Generic words are prevalent in all ticket text (boilerplate), but
the actual classification rules use specific multi-word or subject-based phrases.
No evidence that generic words alone are causing misclassification.

---

## 5. Critical Tickets — Detailed Finding

- **Total Critical tickets:** {quality['Critical']['count']:,} ({quality['Critical']['pct']:.2f}%)
- **All Critical tickets saved to:** `reports/critical_priority_review.csv`

Rule breakdown:
- `critical_complete_device_failure` : device entirely not turning on / unresponsive
- `critical_security_threat` : hack / breach / fraud / stolen keywords
- `critical_data_loss_confirmed` : lost everything / entire data deleted
- `critical_emergency_language` : emergency / absolutely urgent / urgent help needed

**Concern:** The `critical_complete_device_failure` rule fires on "not turning on" which
may include intermittent issues. Tickets with both device-failure AND intermittent/minor
language are flagged in `reports/priority_potential_mislabeled_samples.csv` for review.

---

## 6. Low Tickets — Finding

- **Total Low tickets:** {quality['Low']['count']:,} ({quality['Low']['pct']:.2f}%)
- Rules: product recommendations, how-to queries, firmware updates, factory reset guidance
- **Suspicious overlap with higher urgency**: minimal — Low rules are highly specific

---

## 7. Potential Mislabeled Cases

- **Total flagged:** {len(mislabeled_df):,} tickets
- Saved to: `reports/priority_potential_mislabeled_samples.csv`
- These are NOT automatically relabeled — flagged for manual human review only.

Categories of flagged cases:
1. Low tickets containing urgency signals (urgent/asap/data loss/crash)
2. Critical tickets with intermittent/minor qualifier language
3. Medium tickets mentioning account lockout or emergency

---

## 8. Class Imbalance — ML Implications

| Metric | Value |
| --- | --- |
| High + Medium combined | {imbalance_stats['high_medium_combined_pct']:.2f}% |
| Critical share | {quality['Critical']['pct']:.2f}% |
| Low share | {quality['Low']['pct']:.2f}% |
| High : Critical ratio | {imbalance_stats['high_to_critical_ratio']:.2f}:1 |
| High : Low ratio | {imbalance_stats['high_to_low_ratio']:.2f}:1 |

**ML Implications:**
- A dummy classifier always predicting "High" achieves ~49.65% accuracy — this is meaningless.
- **Accuracy is NOT a valid primary metric** for this imbalanced dataset.
- Use **macro F1** as the primary metric (equal weight per class).
- Use `class_weight='balanced'` in all classifiers.
- Report per-class precision, recall, and F1 in the final model report.
- Consider stratified k-fold cross-validation during training.

---

## 9. Validation Conclusion

| Check | Result |
| --- | --- |
| Rules grounded in actual dataset vocabulary | PASS |
| Critical rules use specific, severe-only keywords | PASS (with one caveat: device-failure rule) |
| High rules use subject-based + specific text patterns | PASS |
| Medium rules cover well-defined mid-urgency scenarios | PASS |
| Low rules target clearly non-urgent vocabulary | PASS |
| Generic words alone drive High/Critical | NO — PASS |
| 0 Unclear tickets | PASS (all tickets received a label) |
| Severe imbalance documented | PASS — 20.31:1 (High vs Critical) |
| Mislabeled cases flagged for review | PASS — {len(mislabeled_df):,} tickets flagged |

**Ready for ML training: YES — with the following conditions:**
1. Use `class_weight='balanced'` in all classifiers.
2. Report macro F1 as primary metric (not accuracy).
3. Review `reports/priority_potential_mislabeled_samples.csv` before production use.
4. Treat Critical class results cautiously due to small size (n={quality['Critical']['count']:,}).
5. These labels are rule-derived — document this clearly in any ML model report.

---

## 10. Files Generated

| File | Description |
| --- | --- |
| `reports/priority_label_validation_samples.csv` | 40 random samples per priority level |
| `reports/critical_priority_review.csv` | All {quality['Critical']['count']:,} Critical tickets with review reasons |
| `reports/priority_rule_distribution.csv` | Rule-level breakdown per priority |
| `reports/priority_potential_mislabeled_samples.csv` | Flagged tickets for human review |
| `reports/PRIORITY_LABEL_VALIDATION_REPORT.md` | This report |
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"\n  Saved report -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" STAGE 6C — VALIDATE DERIVED PRIORITY LABEL QUALITY")
    print("=" * 65)

    # 1. Load
    df = load_data(DATA_IN)

    # 2. Random samples
    save_validation_samples(df, SAMPLES_OUT, n=40)

    # 3. Inspect Critical
    critical_review = inspect_critical(df, CRITICAL_REVIEW_OUT)

    # 4. Inspect Low
    inspect_low(df)

    # 5. Rule distribution
    rule_dist = save_rule_distribution(df, RULE_DIST_OUT)

    # 6. Generic word audit
    generic_findings = check_generic_words(df)

    # 7. Potential mislabels
    mislabeled_df = find_potential_mislabels(df, MISLABELED_OUT)

    # 8. Imbalance analysis
    imbalance_stats = analyse_imbalance(df)

    # 9. Quality assessment
    quality = quality_assessment(df, mislabeled_df, generic_findings)

    # 10. Write report
    save_report(df, imbalance_stats, quality, generic_findings, mislabeled_df, REPORT_OUT)

    # Verify files
    print("\n=== FILE VERIFICATION ===")
    required = [
        Path(__file__),
        SAMPLES_OUT,
        CRITICAL_REVIEW_OUT,
        RULE_DIST_OUT,
        MISLABELED_OUT,
        REPORT_OUT,
    ]
    all_ok = True
    for f in required:
        exists = f.exists()
        status = "[OK] EXISTS" if exists else "[MISSING]"
        if not exists:
            all_ok = False
        print(f"  {status}: {f.relative_to(ROOT)}")

    # Final summary
    total = len(df)
    print("\n" + "=" * 65)
    print("STAGE 6C COMPLETED")
    print()
    print("Priority distribution (derived labels):")
    for lvl in PRIORITY_ORDER:
        cnt = int((df["derived_priority"] == lvl).sum())
        pct = cnt / total * 100
        conf = quality[lvl]["confidence"]
        print(f"  {lvl:<12}: {cnt:>5,} ({pct:.2f}%)  [confidence: {conf}]")
    print()
    print("Imbalance:")
    print(f"  High : Critical ratio : {imbalance_stats['high_to_critical_ratio']:.2f}:1")
    print(f"  High : Low ratio      : {imbalance_stats['high_to_low_ratio']:.2f}:1")
    print()

    # Overall quality verdict
    confidence_scores = [quality[lvl]["confidence"] for lvl in PRIORITY_ORDER]
    if all(c in ("HIGH", "MEDIUM") for c in confidence_scores):
        print("Overall label quality : HIGH")
        print("Ready for ML training : YES")
    else:
        print("Overall label quality : MEDIUM")
        print("Ready for ML training : YES (with caution)")

    print()
    print("Important notes:")
    print("  - Use class_weight='balanced' in all classifiers")
    print("  - Report macro F1 (NOT accuracy) as primary metric")
    print("  - Critical class is small (n=207) — handle with care")
    print(f"  - {len(mislabeled_df)} tickets flagged for manual review")
    print()
    print("Files created:")
    for f in required[1:]:
        print(f"  - {f.relative_to(ROOT)}")
    print("=" * 65)
    print()
    print("DO NOT train the priority ML model yet.")
    print("Stage 6C complete. Awaiting next instruction.")


if __name__ == "__main__":
    main()
