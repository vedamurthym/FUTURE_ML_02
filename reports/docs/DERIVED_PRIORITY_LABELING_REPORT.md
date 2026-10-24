# DERIVED PRIORITY LABELING REPORT
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

| Rule Name | Priority | Description |
| --- | --- | --- |
| `critical_security_threat` | Critical | Explicit security threat: hacking, breach, fraud, stolen credentials |
| `critical_data_loss_confirmed` | Critical | Confirmed data loss or total wipe event |
| `critical_emergency_language` | Critical | Customer uses explicit emergency / critical / urgent-now language |
| `critical_complete_device_failure` | Critical | Device completely dead, unresponsive, will not turn on at all |
| `high_account_access_blocked` | High | Customer is locked out of account / cannot log in |
| `high_account_access_subject` | High | Ticket Subject = account access |
| `high_data_loss_subject` | High | Ticket Subject = data loss |
| `high_data_loss_text` | High | Data loss or deleted files mentioned in ticket text |
| `high_payment_failure` | High | Payment processing failure or transaction error |
| `high_payment_subject` | High | Ticket Subject = payment issue |
| `high_network_blocked` | High | Complete network failure preventing use of product/service |
| `high_network_problem_subject` | High | Ticket Subject = network problem |
| `high_software_crash` | High | Application crash preventing use |
| `high_software_bug_subject` | High | Ticket Subject = software bug |
| `high_hardware_failure` | High | Hardware not charging or other significant hardware fault |
| `high_hardware_issue_subject` | High | Ticket Subject = hardware issue |
| `high_urgent_language` | High | Customer uses urgent / ASAP language (without explicit critical indicators) |
| `medium_refund_request` | Medium | Customer requests a refund |
| `medium_refund_subject` | Medium | Ticket Subject = refund request |
| `medium_cancellation` | Medium | Customer wants to cancel a subscription/order/service |
| `medium_cancellation_subject` | Medium | Ticket Subject = cancellation request |
| `medium_delivery_problem` | Medium | Delivery or order tracking issue |
| `medium_delivery_subject` | Medium | Ticket Subject = delivery problem |
| `medium_installation_support` | Medium | Customer needs help installing or setting up a product |
| `medium_installation_subject` | Medium | Ticket Subject = installation support or product setup |
| `medium_compatibility` | Medium | Product or peripheral compatibility question |
| `medium_compatibility_subject` | Medium | Ticket Subject = peripheral compatibility or product compatibility |
| `medium_display_issue` | Medium | Display or screen problem but product still partially usable |
| `medium_display_subject` | Medium | Ticket Subject = display issue |
| `medium_battery_life` | Medium | Battery draining faster than expected but device still functional |
| `medium_battery_subject` | Medium | Ticket Subject = battery life |
| `medium_general_not_working` | Medium | General 'not working' without specific critical/high indicators |
| `low_product_recommendation` | Low | Customer asks for a product recommendation |
| `low_recommendation_subject` | Low | Ticket Subject = product recommendation |
| `low_general_inquiry` | Low | General information request or how-to question |
| `low_troubleshooting_request` | Low | Customer asks about troubleshooting steps only, no urgency |
| `low_firmware_update` | Low | Firmware/software update question |
| `low_factory_reset` | Low | Customer performed/asking about a factory reset |

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

| Priority | Count | Percentage |
| --- | --- | --- |
| Critical | 207 | 2.44% |
| High | 4,205 | 49.65% |
| Medium | 3,734 | 44.09% |
| Low | 323 | 3.81% |
| Unclear | 0 | 0.00% |

**Total rows:** 8,469
**Labeled rows:** 8,469
**Unclear rows:** 0 (0.00%)

---

## 7. Unclear Cases

Tickets labeled **Unclear** (0, 0.00%) did not match any rule.

Common reasons:
- Highly generic text with no specific urgency or subject keyword
- Synthetic boilerplate sentences with no actionable vocabulary
- Ticket descriptions that are very short or vague

Recommendation: Review a random sample of Unclear tickets to determine whether
additional rules can reasonably classify them, or keep them as a "no-label" set.

---

## 8. Class Imbalance

Among 8,469 labeled tickets:
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
