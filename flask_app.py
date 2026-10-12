import os
import json
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory

from explainer import analyze_caution_indicators
from history import save_analysis_record, get_dashboard_stats, clear_history

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
EVAL_FILE = os.path.join(MODELS_DIR, "evaluation_results.json")

# Load models and vectorizer
vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None

available_models = {}
model_mapping = {
    "Multinomial Naive Bayes": "multinomial_naive_bayes.pkl",
    "Linear Support Vector Machine": "linear_support_vector_machine.pkl",
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl"
}

for name, fname in model_mapping.items():
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        available_models[name] = joblib.load(path)

eval_data = None
if os.path.exists(EVAL_FILE):
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

def run_prediction_pipeline(message_text: str, selected_model_name: str):
    if not message_text.strip() or not vectorizer:
        return None

    if selected_model_name not in available_models:
        selected_model_name = list(available_models.keys())[0]

    clf = available_models[selected_model_name]
    vec = vectorizer.transform([message_text])
    pred_class = int(clf.predict(vec)[0])

    if hasattr(clf, "predict_proba"):
        spam_prob = float(clf.predict_proba(vec)[0][1])
    else:
        spam_prob = 1.0 if pred_class == 1 else 0.0

    analysis = analyze_caution_indicators(message_text, ml_probability=spam_prob)
    is_spam = (pred_class == 1) or (analysis["heuristic_score"] >= 60)

    result_payload = {
        "is_spam": is_spam,
        "prediction_label": "Potential Spam" if is_spam else "Likely Safe",
        "pred_class": pred_class,
        "spam_probability": spam_prob,
        "model_name": selected_model_name,
        "analysis": analysis
    }

    # Record telemetry in history
    save_analysis_record({
        "text": message_text,
        "model": selected_model_name,
        "prediction": result_payload["prediction_label"],
        "confidence": spam_prob,
        "risk_level": analysis["risk_level"],
        "risk_score": analysis["composite_risk_score"],
        "indicators_count": len(analysis["indicators"])
    })

    return result_payload

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message_text = ""
    selected_model = "Multinomial Naive Bayes"

    if request.method == "POST":
        message_text = request.form.get("message", "").strip()
        selected_model = request.form.get("model_choice", "Multinomial Naive Bayes")
        if message_text:
            result = run_prediction_pipeline(message_text, selected_model)

    return render_template(
        "index.html",
        active_page="analyze",
        message=message_text,
        result=result,
        selected_model=selected_model,
        available_models=list(available_models.keys()),
        eval_data=eval_data
    )

@app.route("/analyze", methods=["POST"])
def analyze_post():
    return index()

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or request.form
    text = data.get("message", "").strip()
    model_choice = data.get("model_choice", "Multinomial Naive Bayes")
    if not text:
        return jsonify({"error": "Message text cannot be empty"}), 400
    res = run_prediction_pipeline(text, model_choice)
    return jsonify(res)

@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_stats()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        stats=stats,
        eval_data=eval_data
    )

@app.route("/dashboard/clear", methods=["POST"])
def clear_dashboard_telemetry():
    clear_history()
    return redirect(url_for("dashboard"))

@app.route("/models")
@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        active_page="models",
        eval_data=eval_data
    )

@app.route("/how-it-works")
def how_it_works():
    return render_template(
        "how_it_works.html",
        active_page="how_it_works"
    )

@app.route("/dataset")
def dataset_page():
    return render_template(
        "dataset.html",
        active_page="dataset",
        eval_data=eval_data
    )

@app.route("/live-simulator")
def live_simulator():
    return render_template(
        "live_simulator.html",
        active_page="live_simulator"
    )

@app.route("/research")
def research():
    return render_template(
        "research.html",
        active_page="research",
        eval_data=eval_data
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
