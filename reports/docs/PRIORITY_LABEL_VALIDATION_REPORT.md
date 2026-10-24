# PRIORITY LABEL VALIDATION REPORT
## Stage 6C — Derived Priority Label Quality Assessment

> **⚠️ IMPORTANT**
> These labels are RULE-DERIVED, not human-annotated ground truth.
> This report validates internal rule consistency and logical plausibility.
> It does NOT guarantee real-world correctness.

---

## 1. Overall Label Quality Assessment

**Overall label quality : HIGH — suitable for supervised ML training with balanced weighting**

---

## 2. Priority Distribution

| Priority | Count | % of Total | Confidence |
| --- | --- | --- | --- |
| Critical | 207 | 2.44% | **MEDIUM** |
| High | 4,205 | 49.65% | **HIGH** |
| Medium | 3,734 | 44.09% | **HIGH** |
| Low | 323 | 3.81% | **HIGH** |

**Total rows:** 8,469
**High + Medium combined:** 93.74%
**High : Critical ratio:** 20.31:1
**High : Low ratio:** 13.02:1

---

## 3. Quality Assessment Per Priority Level

| Priority | Confidence | Flagged for Review | Rationale |
| --- | --- | --- | --- |
| Critical | **MEDIUM** | 9 (4.3%) | Rules are specific (hack/breach/fraud/emergency), but 'not turning on' can be ambiguous. ~87% of Cri... |
| High | **HIGH** | 0 (0.0%) | Subject-based High rules are very precise (each subject maps cleanly). high_urgent_language requires... |
| Medium | **HIGH** | 1 (0.0%) | Very clear semantics: refund requests, cancellations, delivery problems, battery issues, compatibili... |
| Low | **HIGH** | 0 (0.0%) | Rules target 'product recommendation', 'how to', 'user manual', 'firmware update', 'factory reset' —... |

---

## 4. Generic Word Audit (Step 6)

The following generic words appear frequently in High/Critical tickets.
These words did NOT drive the classification — they appear in text matched by
specific-phrase rules. No generic word alone triggers High or Critical.

| Generic Word | In High/Critical | % of High+Critical |
| --- | --- | --- |
| `problem` | 1,605 | 36.4% |
| `issue` | 3,900 | 88.4% |
| `help` | 510 | 11.6% |
| `support` | 568 | 12.9% |
| `error` | 483 | 10.9% |
| `need` | 786 | 17.8% |
| `unable` | 600 | 13.6% |

**Finding:** Generic words are prevalent in all ticket text (boilerplate), but
the actual classification rules use specific multi-word or subject-based phrases.
No evidence that generic words alone are causing misclassification.

---

## 5. Critical Tickets — Detailed Finding

- **Total Critical tickets:** 207 (2.44%)
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

- **Total Low tickets:** 323 (3.81%)
- Rules: product recommendations, how-to queries, firmware updates, factory reset guidance
- **Suspicious overlap with higher urgency**: minimal — Low rules are highly specific

---

## 7. Potential Mislabeled Cases

- **Total flagged:** 10 tickets
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
| High + Medium combined | 93.74% |
| Critical share | 2.44% |
| Low share | 3.81% |
| High : Critical ratio | 20.31:1 |
| High : Low ratio | 13.02:1 |

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
| Mislabeled cases flagged for review | PASS — 10 tickets flagged |

**Ready for ML training: YES — with the following conditions:**
1. Use `class_weight='balanced'` in all classifiers.
2. Report macro F1 as primary metric (not accuracy).
3. Review `reports/priority_potential_mislabeled_samples.csv` before production use.
4. Treat Critical class results cautiously due to small size (n=207).
5. These labels are rule-derived — document this clearly in any ML model report.

---

## 10. Files Generated

| File | Description |
| --- | --- |
| `reports/priority_label_validation_samples.csv` | 40 random samples per priority level |
| `reports/critical_priority_review.csv` | All 207 Critical tickets with review reasons |
| `reports/priority_rule_distribution.csv` | Rule-level breakdown per priority |
| `reports/priority_potential_mislabeled_samples.csv` | Flagged tickets for human review |
| `reports/PRIORITY_LABEL_VALIDATION_REPORT.md` | This report |
