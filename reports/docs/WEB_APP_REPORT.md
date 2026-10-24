# WEB APPLICATION REPORT
## Stage 9 — AI Customer Support Decision-Support Web Application

---

## 1. Application Objective

The objective of the **AI Customer Support Ticket Assistant** is to provide a clean, professional,
user-facing dashboard for support agents to triage incoming customer support requests. It automates:
- **Category Classification:** Sorting tickets into one of seven operational categories.
- **Priority Classification:** Assigning a priority tier (*Critical*, *High*, *Medium*, *Low*) to optimize response queues.
- **Decision-Support:** Recommending operational actions and emphasizing when human verification is required (especially for Critical outages/security threats).

---

## 2. Architecture

The web application is built on the following stack:
- **Frontend / Interface:** Flask (Python) serving a custom **HTML/CSS/JavaScript** single-page dashboard with animated sidebar, tabbed interface, drag-and-drop CSV upload, and real-time prediction display.
- **Inference Pipeline:** Reuses prediction functions (`predict_ticket`, `predict_tickets`) directly from `src/08_predict_ticket.py` to prevent code duplication.
- **Machine Learning Models:**
  - Category Model: Deployed from `models/derived_category_model.pkl` (fitted with Category Vectorizer `models/derived_category_tfidf_vectorizer.pkl`).
  - Priority Model: Deployed from `models/priority_model.pkl` (fitted with Priority Vectorizer `models/priority_tfidf_vectorizer.pkl`).

---

## 3. User Workflow

The application supports two distinct triage workflows:
1. **Real-time Single Ticket Analysis:**
   - The user inputs or clicks an example ticket in the sidebar to populate the input text area.
   - Click "Analyze Ticket" to run predictions.
   - View predicted category and priority alongside their relative confidence decision scores.
   - Display recommended actions and human review disclaimers.
2. **Batch CSV Prediction:**
   - Upload a CSV containing a column named exactly `ticket_text`.
   - The app processes all entries in bulk using batch inference (`predict_tickets`).
   - Displays a preview table of predictions.
   - Click "Download Predictions CSV" to save the labeled data locally.

---

## 4. Category Prediction

Classifies ticket content into one of seven possible categories:
- *Account & Access*
- *Billing & Payment*
- *Cancellation Request*
- *Delivery & Order Inquiry*
- *Product Inquiry*
- *Refund Request*
- *Technical Issue*

A relative confidence decision score (raw decision function value from the multi-class LinearSVC) is shown for transparency.

---

## 5. Priority Prediction

Classifies ticket urgency into four possible levels:
- *Critical*
- *High*
- *Medium*
- *Low*

Critical predictions are highlighted in a red warning card to ensure support agents notice them immediately.

---

## 6. Decision-Support Recommendations

Operational guidance displayed to the user based on predicted priority levels:
- **Critical:** "Immediate human review recommended. Route to the Emergency Response Team immediately."
- **High:** "Prioritize for prompt support response. Target response within 2-4 hours."
- **Medium:** "Handle through normal support queue. Target response within 1 business day."
- **Low:** "Can be handled through the standard support workflow. Auto-respond with relevant self-service articles if possible."

---

## 7. Batch Prediction

Enables bulk ticket routing by uploading a CSV. Input validation is performed to ensure the `ticket_text` column is present. Row-level empty values are handled safely, returning a validation failure status without crashing the process. The downloaded output contains predicted labels, relative confidence scores, and any validation messages.

---

## 8. Deployed Model Metrics

Evaluated performance markers:

| Model | Algorithm | Accuracy | Macro F1-Score | Special Metric |
| --- | --- | --- | --- | --- |
| **Category** | Linear SVM (Balanced) | 99.82% | 99.79% | — |
| **Priority** | Linear SVM (Balanced) | 98.41% | 94.82% | **82.93%** (Critical Recall) |

---

## 9. Human-Review Warnings

- **AI-assisted Triage Alert:** The interface clearly advises: *"Always review predictions before taking customer-facing action."*
- **Critical Class Warning:** Emphasizes that because *Critical* tickets represent only 2.44% of the training dataset, predictions in this class must be manually validated by a human agent before dispatching emergency workflows.

---

## 10. Limitations

1. **Synthetic data bias:** The underlying Kaggle dataset contains highly boilerplate syntax; performance on highly conversational real-world customer phrasing may degrade.
2. **Rule-derived labels:** The predictions model keyword/rule mappings rather than human-curated intent.
3. **No probability calibration:** Decision scores represent raw distance to the decision boundary (OVR LinearSVC), not calibrated probabilities.

---

## 11. How to Run the Application

Run the Flask application from the `support-ticket-classification` root folder:

```bash
python app.py
```

Open your browser at `http://localhost:5000`.

Ensure all packages in `requirements.txt` are installed in your active environment:

```bash
pip install -r requirements.txt
```
