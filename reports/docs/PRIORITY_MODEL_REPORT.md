# PRIORITY MODEL TRAINING REPORT
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
| Total rows | 8,469 |
| Training rows | 6,775 (80%) |
| Testing rows | 1,694 (20%) |
| Split method | Stratified 80/20 (random_state=42) |

**Training Set**
| Priority | Count | % |
| --- | --- | --- |
| Critical | 166 | 2.45% |
| High | 3,364 | 49.65% |
| Medium | 2,987 | 44.09% |
| Low | 258 | 3.81% |

**Testing Set**
| Priority | Count | % |
| --- | --- | --- |
| Critical | 41 | 2.42% |
| High | 841 | 49.65% |
| Medium | 747 | 44.10% |
| Low | 65 | 3.84% |

---

## 5. Class Imbalance

| Metric | Value |
| --- | --- |
| High + Medium combined | 93.7% (train) |
| Critical share | 2.45% (train) |
| Low share | 3.81% (train) |
| High : Critical ratio | 20.3:1 |

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
| max_features | 50,000 |
| min_df | 2 |
| sublinear_tf | True |
| Features used | 15,639 |

---

## 8. Models Tested

| Model | CW | Acc | W-F1 | **Macro F1** | Crit-R | Crit-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Linear SVM | balanced | 0.9841 | 0.9841 | **0.9482** | 0.8293 | 0.9067 |
| Linear SVM | none | 0.9841 | 0.9840 | **0.9461** | 0.8049 | 0.8919 |
| Logistic Regression | balanced | 0.9787 | 0.9794 | **0.9348** | 0.8293 | 0.9067 |
| Logistic Regression | none | 0.9799 | 0.9793 | **0.9304** | 0.8049 | 0.8919 |
| Multinomial Naive Bayes | none | 0.8058 | 0.7839 | **0.4695** | 0.1220 | 0.2174 |

*CW = Class Weight; W-F1 = Weighted F1; Crit-R = Critical Recall*

---

## 9. Best Model

**Selected: Linear SVM [class_weight=balanced]**

Selection criteria (in order):
1. Macro F1 = **0.9482** (primary — equal weight per class)
2. Critical F1 = **0.9067** (missing critical tickets is costly)
3. Weighted F1 = **0.9841** (overall quality)
4. Accuracy excluded as primary criterion (misleading with 49.65% majority class)

---

## 10. Per-Class Performance (Best Model)

| Priority | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Critical | 1.0000 | 0.8293 | 0.9067 | 41 |
| High | 0.9917 | 0.9929 | 0.9923 | 841 |
| Medium | 0.9892 | 0.9826 | 0.9859 | 747 |
| Low | 0.8421 | 0.9846 | 0.9078 | 65 |

**Critical Recall = 0.8293**
Operationally, failing to detect a Critical ticket is more costly than any other
confusion type. This metric should be monitored closely in production.

---

## 11. Error Analysis

Total misclassified: 27 / 1,694
Error rate: 1.59%

Most common confusion pairs:

| Actual → Predicted | Count |
| --- | --- |
| Medium → Low | 12 |
| Critical → High | 6 |
| High → Medium | 6 |
| Critical → Medium | 1 |
| Low → Medium | 1 |
| Medium → High | 1 |

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

The best model — **Linear SVM [class_weight=balanced]** — achieves:

| Metric | Value |
| --- | --- |
| Macro F1 | 0.9482 |
| Weighted F1 | 0.9841 |
| Accuracy | 0.9841 |
| Critical Recall | 0.8293 |
| Critical F1 | 0.9067 |

The model is suitable for internal ticket routing provided:
- `class_weight='balanced'` is applied
- Results are evaluated using macro F1, not accuracy
- Critical-class predictions are validated by human agents
- Labels are documented as rule-derived, not human-annotated
