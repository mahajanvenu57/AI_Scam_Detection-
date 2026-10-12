import os
import sys
import joblib
from explainer import analyze_caution_indicators

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def test_inference():
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        print("Model files not found!")
        return

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)

    test_messages = [
        "URGENT: Your bank account will be blocked today. Click http://secure-bank-verify.net immediately to update KYC details.",
        "Congratulations! You won $5000 in our lucky draw! Call 09061701461 to claim your reward.",
        "Hey, can you please send me the notes from today's lecture? Thanks!",
        "Your Amazon package has been delivered to your doorstep. Have a great day!"
    ]

    print("\n--- Testing Model Predictions and Explainability ---\n")
    for msg in test_messages:
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
        expl = analyze_caution_indicators(msg, ml_probability=prob)

        status = "SPAM / SCAM" if pred == 1 or expl['heuristic_score'] >= 60 else "LEGITIMATE (HAM)"
        print(f"Message: {msg}")
        print(f"Prediction: {status} | Spam Probability: {prob:.2%}")
        print(f"Composite Risk Score: {expl['composite_risk_score']}/100")
        print(f"Indicators Detected: {len(expl['indicators'])}")
        for ind in expl['indicators']:
            desc = ind.get('description', ind.get('details', ''))
            print(f"  - [{ind['level']}] {ind.get('name', ind.get('category'))}: {desc}")
        print("-" * 60)

if __name__ == "__main__":
    test_inference()
