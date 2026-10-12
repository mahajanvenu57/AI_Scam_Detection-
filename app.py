import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from explainer import analyze_caution_indicators
from history import save_analysis_record, get_dashboard_stats

# Page configuration
st.set_page_config(
    page_title="SCAMSHIELD AI | SMS Scam Detection & Risk Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_FILE = os.path.join(MODELS_DIR, "evaluation_results.json")

# Dark Cybersecurity CSS Styling
st.markdown("""
<style>
    /* Dark Theme Cyber Overrides */
    .stApp {
        background-color: #070b13;
        color: #f8fafc;
    }
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    .badge-cyber {
        display: inline-block;
        background: rgba(6, 182, 212, 0.12);
        color: #06b6d4;
        border: 1px solid rgba(6, 182, 212, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
        font-family: monospace;
    }
    .cyber-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .danger-banner {
        background: rgba(225, 29, 72, 0.15);
        border: 1px solid rgba(225, 29, 72, 0.45);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .safe-banner {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .caution-pill {
        background: #141e33;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        return None, None, None

    vectorizer = joblib.load(vectorizer_path)
    
    models = {}
    model_files = {
        "Multinomial Naive Bayes": "multinomial_naive_bayes.pkl",
        "Linear Support Vector Machine": "linear_support_vector_machine.pkl",
        "Random Forest": "random_forest.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl"
    }
    for m_name, f_name in model_files.items():
        f_path = os.path.join(MODELS_DIR, f_name)
        if os.path.exists(f_path):
            models[m_name] = joblib.load(f_path)
            
    eval_data = None
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    return vectorizer, models, eval_data

vectorizer, models, eval_data = load_artifacts()

# Sidebar
st.sidebar.markdown("### 🛡️ SCAMSHIELD AI")
st.sidebar.markdown("**Project Code:** MCARP31")
st.sidebar.markdown("**Institution:** SYMCA Semester III Research Project")
st.sidebar.divider()
st.sidebar.markdown("### 👥 Student Researchers")
st.sidebar.markdown("• **Harshi Anand Naik** (Roll: 24)")
st.sidebar.markdown("• **Venu Mahajan** (Roll: 15)")
st.sidebar.markdown("• **Sayali Shelake** (Roll: 41)")
st.sidebar.divider()

if eval_data:
    st.sidebar.success(f"✅ Baseline: **{eval_data.get('best_model', 'Active')}**")
    st.sidebar.caption(f"Dataset: UCI SMS Collection ({eval_data.get('dataset_total', 5574)} samples)")
else:
    st.sidebar.warning("⚠️ Run `python train_models.py` to compile models.")

# Header
st.markdown('<div class="badge-cyber">SCAMSHIELD AI</div><div class="badge-cyber">MCARP31 • RESEARCH PROJECT</div>', unsafe_allow_mode=True)
st.markdown('<div class="main-title">SCAMSHIELD AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Based SMS Scam Detection & Risk Analysis</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Scan SMS",
    "📊 Telemetry Dashboard",
    "📈 Model Evaluation",
    "⚙️ How It Works",
    "📁 Dataset Specs",
    "🎓 Research Methodology"
])

# ----------------- TAB 1: Scan SMS -----------------
with tab1:
    col_in, col_res = st.columns([7, 5])

    with col_in:
        st.subheader("Message Scanner")
        
        # Sample presets
        presets = {
            "— Choose a Preset Sample —": "",
            "🚨 Bank KYC Suspension Alert (Scam)": "URGENT: Your SBI account has been suspended! Please update your KYC immediately at http://sbi-kyc-update.xyz/login within 24 hours to avoid permanent deactivation.",
            "🎁 £1M Lottery Cash Prize (Scam)": "Congratulations! You have won a cash prize of £1,000,000 in the UK National Lottery. Claim your prize now by calling 09061701461. Offer valid for 24h only!",
            "⚡ Electricity Disconnection Threat (Scam)": "Dear Consumer, your electricity power will be disconnected tonight at 9:30 PM due to unpaid bill. Call electricity officer immediately at 9876543210.",
            "💬 College Study Group Chat (Ham)": "Hey Harshi, are we meeting at the college library at 3 PM for the MCARP31 presentation prep? Let me know!",
            "📦 Official Amazon Delivery Update (Ham)": "Your order #849201 from Amazon has been dispatched. Track your delivery inside the official Amazon app. Arriving tomorrow."
        }

        selected_preset = st.selectbox("Quick Test Samples:", list(presets.keys()))
        default_val = presets[selected_preset] if selected_preset != "— Choose a Preset Sample —" else ""

        user_sms = st.text_area(
            "SMS Text Input:",
            value=default_val,
            height=140,
            placeholder="Paste or enter SMS message here..."
        )
        st.caption(f"Length: {len(user_sms)} characters")

        col_m, col_b = st.columns([2, 1])
        with col_m:
            chosen_model = st.selectbox("Classifier Algorithm:", list(models.keys()) if models else ["Multinomial Naive Bayes"])
        with col_b:
            st.write("")
            st.write("")
            scan_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)

        st.caption("🔒 **Privacy Guarantee:** Messages are processed strictly locally. No content is transmitted to third parties.")

    with col_res:
        st.subheader("Risk Telemetry")
        if scan_clicked:
            if not user_sms.strip():
                st.error("Please enter a non-empty SMS message.")
            elif not vectorizer or not models or chosen_model not in models:
                st.error("Model artifacts missing. Run `python train_models.py` first.")
            else:
                with st.spinner("Analyzing message against models & caution heuristics..."):
                    clf = models[chosen_model]
                    tfidf_vec = vectorizer.transform([user_sms])
                    pred_class = int(clf.predict(tfidf_vec)[0])
                    prob = float(clf.predict_proba(tfidf_vec)[0][1]) if hasattr(clf, "predict_proba") else (1.0 if pred_class == 1 else 0.0)

                    analysis = analyze_caution_indicators(user_sms, ml_probability=prob)
                    is_spam = (pred_class == 1) or (analysis["heuristic_score"] >= 60)

                    # Save record in telemetry history
                    save_analysis_record({
                        "text": user_sms,
                        "model": chosen_model,
                        "prediction": "Potential Spam" if is_spam else "Likely Safe",
                        "confidence": prob,
                        "risk_level": analysis["risk_level"],
                        "risk_score": analysis["composite_risk_score"],
                        "indicators_count": len(analysis["indicators"])
                    })

                    # Display banner
                    if is_spam:
                        st.markdown(f"""
                        <div class="danger-banner">
                            <h4 style="color: #f43f5e; margin:0;"><i class="fas fa-triangle-exclamation"></i> POTENTIAL SPAM / SCAM</h4>
                            <small style="color: #cbd5e1;">Identified as fraudulent, smishing, or malicious communication.</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-banner">
                            <h4 style="color: #34d399; margin:0;"><i class="fas fa-circle-check"></i> LIKELY SAFE (HAM)</h4>
                            <small style="color: #cbd5e1;">No malicious or coercive smishing indicators detected.</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Classifier", chosen_model.split()[0])
                    c2.metric("Spam Confidence", f"{prob*100:.1f}%")
                    c3.metric("Risk Level", f"{analysis['risk_level']} ({analysis['composite_risk_score']}%)")

                    # Caution Indicators
                    st.markdown("##### ⚠️ Explainable Caution Indicators:")
                    if analysis["indicators"]:
                        for ind in analysis["indicators"]:
                            st.markdown(f"""
                            <div class="caution-pill">
                                <div style="display:flex; justify-content:space-between;">
                                    <strong>{ind['icon']} {ind['name']}</strong>
                                    <span style="font-size:0.75rem; color:#f43f5e; font-weight:bold;">{ind['level']}</span>
                                </div>
                                <div style="font-size:0.83rem; color:#94a3b8; margin-top:4px;">{ind['description']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No suspicious caution indicators detected.")

                    # Recommendations
                    st.markdown("##### 🛡️ Safety Guidance:")
                    for g in analysis["safety_guidance"]:
                        st.caption(f"• {g}")

                    st.markdown(f"<div style='font-size:0.72rem; color:#64748b; margin-top:10px;'>{analysis['disclaimer']}</div>", unsafe_allow_html=True)
        else:
            st.info("Enter or select an SMS message on the left to inspect prediction confidence and explainable indicators.")

# ----------------- TAB 2: Telemetry Dashboard -----------------
with tab2:
    st.subheader("Security Operations Dashboard")
    stats = get_dashboard_stats()

    if not stats["has_history"]:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h4 style="color: #94a3b8;">No messages analyzed yet.</h4>
            <p style="color: #64748b; font-size: 0.9rem;">Run messages through the scanner tab to populate telemetry.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Total Analyzed", stats["total_analyzed"])
        col_d2.metric("Likely Safe (Ham)", stats["ham_count"])
        col_d3.metric("Potential Spam", stats["spam_count"])
        col_d4.metric("Severe Threats", stats["risk_distribution"]["Severe"])

        st.divider()
        st.markdown("#### Recent Analysis Log")
        hist_df = pd.DataFrame(stats["recent_analyses"])
        if not hist_df.empty:
            st.dataframe(hist_df[["timestamp", "text", "model", "prediction", "risk_level", "risk_score"]], use_container_width=True)

# ----------------- TAB 3: Model Evaluation -----------------
with tab3:
    st.subheader("Model Evaluation & Performance Benchmark")
    st.caption("Corresponds to Section 8.3 of the SYMCA Research Paper, populated from genuine Scikit-learn execution.")

    if eval_data and "models_evaluation" in eval_data:
        table_rows = []
        for m_name, m_metrics in eval_data["models_evaluation"].items():
            table_rows.append({
                "Model": m_name,
                "Accuracy": f"{m_metrics['accuracy']:.4f}",
                "Precision": f"{m_metrics['precision']:.4f}",
                "Recall": f"{m_metrics['recall']:.4f}",
                "F1-score": f"{m_metrics['f1_score']:.4f}",
                "ROC-AUC": f"{m_metrics['roc_auc']:.4f}",
                "5-Fold CV F1": f"{m_metrics['cv_f1_mean']:.4f}",
                "False Positive Rate": f"{m_metrics['false_positive_rate']:.4f}",
                "False Negative Rate": f"{m_metrics['false_negative_rate']:.4f}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        col_c1, col_c2 = st.columns([7, 5])
        with col_c1:
            p_comp = os.path.join(ASSETS_DIR, "model_comparison.png")
            if os.path.exists(p_comp):
                st.image(p_comp, caption="Comparative Performance Across Candidate Models", use_container_width=True)
        with col_c2:
            p_cm = os.path.join(ASSETS_DIR, "best_confusion_matrix.png")
            if os.path.exists(p_cm):
                st.image(p_cm, caption=f"Confusion Matrix ({eval_data['best_model']})", use_container_width=True)

        col_err1, col_err2 = st.columns(2)
        with col_err1:
            st.markdown("##### ⚠️ False Positive Explanation:")
            st.caption("**False Positive = legitimate/ham message incorrectly classified as spam.** Causes user disruption or quarantine of normal SMS.")
        with col_err2:
            st.markdown("##### 🚨 False Negative Explanation:")
            st.caption("**False Negative = spam message incorrectly classified as ham.** Severe cybersecurity risk; exposes user to malicious links.")

# ----------------- TAB 4: How It Works -----------------
with tab4:
    st.subheader("System Architecture & Workflow Pipeline")
    st.markdown("""
    ```
    SMS Input
       ↓
    Text Preprocessing (Case normalization, punctuation handling, whitespace strip)
       ↓
    TF-IDF Feature Extraction (Unigram & Bigram sublinear TF-IDF, 5,000 dimensions)
       ↓
    Machine Learning Model (Multinomial NB / Linear SVM / Random Forest / Logistic Regression / Decision Tree)
       ↓
    Prediction & Probability (Binary classification + Calibrated posterior spam probability)
       ↓
    Risk Interpretation (10-point heuristic security engine scanning urgency, lures, URLs, shortcodes)
       ↓
    Explainable Result (Transparent caution badges, composite risk score 0–100%, safety guidelines)
    ```
    """)
    st.markdown("Data Leakage Prevention: The TF-IDF vectorizer is fitted strictly on training data (80% split) and frozen.")

# ----------------- TAB 5: Dataset Specs -----------------
with tab5:
    st.subheader("Dataset Specifications")
    st.markdown("""
    - **Dataset:** SMS Spam Collection
    - **Source:** UCI Machine Learning Repository
    - **Instances:** 5,574 (4,827 Ham, 747 Spam)
    - **Labels:** `ham` / `spam`
    - **Data Type:** Raw SMS text
    - **Missing Values:** 0
    """)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        p_dist = os.path.join(ASSETS_DIR, "class_distribution.png")
        if os.path.exists(p_dist):
            st.image(p_dist, caption="Class Balance: Ham (86.6%) vs Spam (13.4%)", use_container_width=True)
    with col_d2:
        p_len = os.path.join(ASSETS_DIR, "length_distribution.png")
        if os.path.exists(p_len):
            st.image(p_len, caption="Message Length Distribution", use_container_width=True)

    st.warning("⚠️ **Dataset Boundary:** The dataset does NOT contain sender identity, location, device information, click behavior, or verified financial-loss labels.")

# ----------------- TAB 6: Research Methodology -----------------
with tab6:
    st.subheader("Academic Research & Methodology (MCARP31)")
    st.markdown("""
    **Project Title:** AI-Based Scam Detection  
    **Authors:** Harshi Anand Naik (24), Venu Mahajan (15), Sayali Shelake (41)  
    **Program:** SYMCA Semester III Research Presentation  

    ---
    - **Problem Definition:** Escalating mobile SMS phishing and fraudulent financial lures bypass traditional keyword filters due to abbreviated language.
    - **Methodology:** Supervised machine learning with sublinear TF-IDF feature representation and transparent Explainable AI caution indicators.
    - **Algorithms:** Multinomial Naive Bayes, Linear Support Vector Machine, Logistic Regression, Decision Tree, Random Forest.
    - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, 5-Fold Cross Validation, FPR, FNR.
    - **Ethical & Safety Safeguards:** Strict use of public anonymized CC BY 4.0 data; 100% local inference with zero third-party data transmission; clear assistive advisory positioning.
    """)
