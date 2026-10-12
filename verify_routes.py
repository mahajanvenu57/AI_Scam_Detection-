import os
import json
from flask_app import app

def test_routes_and_inference():
    client = app.test_client()
    print("=" * 60)
    print("VERIFYING SCAMSHIELD AI FLASK APPLICATION & ROUTES")
    print("=" * 60)

    # 1. Test Home / Analyze Page
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200 on /, got {res.status_code}"
    assert b"AI-Based SMS Scam Detection" in res.data or b"SCAMSHIELD" in res.data
    print("[OK] [1/8] GET / loads successfully with academic project branding.")

    # 2. Test Empty Input Handling
    res = client.post("/", data={"message": "", "model_choice": "Multinomial Naive Bayes"})
    assert res.status_code == 200, f"Expected 200 on empty post, got {res.status_code}"
    assert b"No Message Analyzed Yet" in res.data or b"Awaiting Message Input" in res.data
    print("[OK] [2/8] Empty/invalid input handled safely.")

    # 3. Test Real Inference (Spam Scam)
    scam_msg = "URGENT: Your SBI account has been suspended! Please update your KYC immediately at http://sbi-kyc.xyz"
    res = client.post("/", data={"message": scam_msg, "model_choice": "Multinomial Naive Bayes"})
    assert res.status_code == 200
    assert b"POTENTIAL SPAM" in res.data or b"Potential Spam" in res.data
    assert b"Detected Caution Indicators" in res.data or b"Explainable Risk Indicators" in res.data
    print("[OK] [3/8] POST / with scam message generates real prediction & explainable indicators.")

    # 4. Test Real Inference (Ham Message)
    ham_msg = "Hey Harshi, please remember to bring the project files tomorrow for presentation."
    res = client.post("/", data={"message": ham_msg, "model_choice": "Multinomial Naive Bayes"})
    assert res.status_code == 200
    assert b"LIKELY LEGITIMATE" in res.data or b"LIKELY SAFE" in res.data or b"Likely Safe" in res.data
    print("[OK] [4/8] POST / with legitimate message generates Likely Safe verdict.")

    # 5. Test JSON API Endpoint
    res = client.post("/api/analyze", json={"message": scam_msg, "model_choice": "Linear Support Vector Machine"})
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "is_spam" in data
    assert "spam_probability" in data
    assert "analysis" in data
    assert len(data["analysis"]["indicators"]) > 0
    print(f"[OK] [5/8] POST /api/analyze returned valid JSON: Risk Score={data['analysis']['composite_risk_score']}%, Indicators={len(data['analysis']['indicators'])}.")

    # 6. Test Dashboard Page
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert b"Dashboard" in res.data
    assert b"Total Analyzed" in res.data
    print("[OK] [6/8] GET /dashboard loads telemetry and metrics dynamically.")

    # 7. Test Model Performance Page
    res = client.get("/models")
    assert res.status_code == 200
    assert b"Table 1" in res.data
    assert b"0.9839" in res.data, "Calculated metric 0.9839 missing from table"
    assert b"Multinomial Naive Bayes" in res.data
    assert b"Linear Support Vector Machine" in res.data
    assert b"Random Forest" in res.data
    print("[OK] [7/8] GET /models displays dynamic metrics table with real calculated values.")

    # 8. Test How It Works, Dataset, and Research Pages
    res_how = client.get("/how-it-works")
    assert res_how.status_code == 200

    res_sim = client.get("/live-simulator")
    assert res_sim.status_code == 200

    res_ds = client.get("/dataset")
    assert res_ds.status_code == 200 and b"5,574" in res_ds.data

    res_res = client.get("/research")
    assert res_res.status_code == 200 and b"MCARP31" in res_res.data and b"Harshi Anand Naik" in res_res.data

    # Check asset serving
    res_asset = client.get("/assets/model_comparison.png")
    assert res_asset.status_code == 200 and res_asset.content_type == "image/png"

    print("[OK] [8/8] Static assets, /how-it-works, /dataset, and /research loaded successfully.")
    print("=" * 60)
    print("ALL SCAMSHIELD AI SYSTEM VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_routes_and_inference()
