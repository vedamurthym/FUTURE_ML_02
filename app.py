"""
AI Customer Support Ticket Assistant
=====================================
Flask web application for real-time ticket classification and triage.
Reuses inference functions from src/08_predict_ticket.py.

Usage:
  python app.py
  Then open http://localhost:5000 in your browser.
"""

import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import importlib
try:
    predict_ticket_mod = importlib.import_module("src.08_predict_ticket")
    load_prediction_pipeline = predict_ticket_mod.load_prediction_pipeline
    predict_ticket           = predict_ticket_mod.predict_ticket
    predict_tickets          = predict_ticket_mod.predict_tickets
    CAT_MODEL_PATH           = predict_ticket_mod.CAT_MODEL_PATH
    CAT_VEC_PATH             = predict_ticket_mod.CAT_VEC_PATH
    PRIO_MODEL_PATH          = predict_ticket_mod.PRIO_MODEL_PATH
    PRIO_VEC_PATH            = predict_ticket_mod.PRIO_VEC_PATH
except ImportError as e:
    print(f"[FATAL] Failed to import prediction pipeline: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LOAD MODELS AT STARTUP
# ---------------------------------------------------------------------------
print("Loading ML models...")
missing = []
for path, name in [
    (CAT_MODEL_PATH,  "Category Model (derived_category_model.pkl)"),
    (CAT_VEC_PATH,    "Category Vectorizer (derived_category_tfidf_vectorizer.pkl)"),
    (PRIO_MODEL_PATH, "Priority Model (priority_model.pkl)"),
    (PRIO_VEC_PATH,   "Priority Vectorizer (priority_tfidf_vectorizer.pkl)"),
]:
    if not path.exists():
        missing.append(name)

if missing:
    print("[ERROR] Missing model artifacts:")
    for m in missing:
        print(f"  - {m}")
    print("Please run Stage 5D and Stage 6D to generate them.")
    sys.exit(1)

cat_model, cat_vec, prio_model, prio_vec = load_prediction_pipeline()
print("All models loaded successfully.")

# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Single ticket prediction endpoint."""
    body = request.get_json(force=True, silent=True) or {}
    ticket_text = body.get("ticket_text", "")
    t_start = time.time()
    res = predict_ticket(ticket_text, cat_model, cat_vec, prio_model, prio_vec)
    latency_ms = (time.time() - t_start) * 1000
    if res["error"]:
        return jsonify({"error": res["error"]}), 400
    return jsonify({
        "category":                res["category"],
        "category_decision_score": res["category_decision_score"],
        "priority":                res["priority"],
        "priority_decision_score": res["priority_decision_score"],
        "latency_ms":              round(latency_ms, 2),
    })


@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    """Batch CSV prediction endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted."}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    possible_cols = ["ticket_text", "Ticket Description", "text", "description", "ticket", "Ticket Subject"]
    detected_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
    texts = df[detected_col].fillna("").astype(str).tolist()

    t_start = time.time()
    batch_res = predict_tickets(texts, cat_model, cat_vec, prio_model, prio_vec)
    batch_time = time.time() - t_start

    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    rows = []
    for i, r in enumerate(batch_res):
        prio = r.get("priority") or "Medium"
        if prio in counts:
            counts[prio] += 1
        rows.append({
            "ticket_text":        texts[i],
            "predicted_category": r.get("category", ""),
            "predicted_priority": prio,
            "category_score":     round(r.get("category_decision_score") or 0.0, 4),
            "priority_score":     round(r.get("priority_decision_score") or 0.0, 4),
        })

    return jsonify({
        "total":      len(rows),
        "batch_time": round(batch_time, 3),
        "counts":     counts,
        "rows":       rows,
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Customer Support Intelligence Hub")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)
