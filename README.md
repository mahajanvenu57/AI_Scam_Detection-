# 🛡️ SCAMSHIELD AI
### AI-Based SMS Scam Detection & Risk Analysis

> **Academic Research Project:** MCARP31  
> **Course:** Master of Computer Applications (SYMCA Semester III)  
> **Institution:** Research Project Presentation Guidelines  

### 👥 Research Team
- **Harshi Anand Naik** (Roll No: 24)
- **Venu Mahajan** (Roll No: 15)
- **Sayali Shelake** (Roll No: 41)

---

## 📌 Project Overview
**SCAMSHIELD AI** is an explainable artificial intelligence (XAI) and machine learning cybersecurity platform designed to classify mobile SMS communications into legitimate (**Ham**) or fraudulent (**Spam / Phishing**). 

Trained on the peer-reviewed **UCI SMS Spam Collection (5,574 instances)**, the system combines **5 classical supervised algorithms** with a **10-point heuristic risk engine** to output not just a black-box label, but transparent, actionable caution indicators (urgency, fake rewards, banking baits, credential theft, smishing URLs, and contact shortcodes).

---

## 🚀 Key Features

1. **Multi-Model Benchmark (Section 8.3):**
   - Multinomial Naive Bayes (Baseline Champion, 98.39% Accuracy, 0.9375 F1)
   - Linear Support Vector Machine (98.39% Accuracy, 0.9375 F1)
   - Random Forest (100% Precision, 97.76% Accuracy)
   - Logistic Regression (100% Precision, 97.67% Accuracy)
   - Decision Tree (95.25% Accuracy, 0.8179 F1)
2. **Explainable AI (XAI) Security Engine:**
   - 10 caution indicator categories scanning for time pressure, coercion, prize claims, OTP requests, links, phone patterns, and excessive numbers.
   - Dynamic Risk Levels: `Low`, `Elevated`, `High`, `Severe` (Composite Risk Score 0–100%).
3. **Cybersecurity Web Portals:**
   - **Flask Multi-Page Application (`flask_app.py`):** Real-time message scanner, live telemetry dashboard, dynamic benchmark tables, visual workflow, dataset documentation, and viva defense page.
   - **Streamlit Interactive Dashboard (`app.py`):** Dual-support demonstration interface.
4. **Reproducible Research Notebook:**
   - `notebooks/scam_detection_analysis.ipynb` documenting all 21 research workflow steps with genuine execution outputs.

---

## 📊 Populated Research Benchmark (Table 1: Section 8.3)

*Evaluated on an 80/20 Stratified Train-Test Split (Seed = 42) from UCI SMS Spam Collection:*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** ⭐ | **0.9839** | **0.9712** | **0.9060** | **0.9375** | **0.9915** | **0.9428** | **0.0041** | **0.0940** |
| **Linear SVM** | **0.9839** | **0.9712** | **0.9060** | **0.9375** | **0.9914** | **0.9326** | **0.0041** | **0.0940** |
| **Random Forest** | 0.9776 | 1.0000 | 0.8322 | 0.9084 | 0.9877 | 0.9161 | 0.0000 | 0.1678 |
| **Logistic Regression** | 0.9767 | 1.0000 | 0.8255 | 0.9044 | 0.9891 | 0.7743 | 0.0000 | 0.1745 |
| **Decision Tree** | 0.9525 | 0.8380 | 0.7987 | 0.8179 | 0.8631 | 0.8486 | 0.0238 | 0.2013 |

*False Positive Rate (FPR) = FP / (FP + TN) | False Negative Rate (FNR) = FN / (FN + TP)*

---

## 📂 Project Architecture
```
ai_scam_detection/
├── assets/                          # Generated high-resolution evaluation plots
│   ├── best_confusion_matrix.png
│   ├── class_distribution.png
│   ├── length_distribution.png
│   └── model_comparison.png
├── data/                            # Verified UCI benchmark data
│   ├── SMSSpamCollection
│   ├── sms_spam_collection.csv
│   └── analysis_history.json        # Dynamic telemetry log
├── models/                          # Serialized Scikit-learn pipelines & JSON metrics
│   ├── best_model.pkl
│   ├── decision_tree.pkl
│   ├── evaluation_results.json
│   ├── linear_support_vector_machine.pkl
│   ├── logistic_regression.pkl
│   ├── multinomial_naive_bayes.pkl
│   ├── random_forest.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── scam_detection_analysis.ipynb # 21-step reproducible research notebook
├── templates/                       # Dark cybersecurity Bootstrap 5 templates
│   ├── base.html                    # Common layout, header & safety footer
│   ├── index.html                   # Main analysis & real-time scanner page
│   ├── dashboard.html               # Security operations telemetry dashboard
│   ├── performance.html             # Model evaluation & benchmark table
│   ├── how_it_works.html            # Visual system pipeline & TF-IDF math
│   ├── dataset.html                 # Dataset specifications & boundaries
│   └── research.html                # Academic synopsis & viva documentation
├── app.py                           # Streamlit multi-tab application
├── flask_app.py                     # Main Flask web application
├── dataset.py                       # Dataset loader & preprocessor
├── explainer.py                     # Explainable AI & 10 caution indicator engine
├── history.py                       # Session telemetry manager
├── train_models.py                  # Model training & evaluation pipeline
├── test_system.py                   # Automated CLI test suite
├── build_notebook.py                # Notebook builder script
├── requirements.txt                 # Python package dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # Documentation
```

---

## ⚡ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train and Evaluate Models
To regenerate models, run 5-fold cross-validation, and recreate plots:
```bash
python train_models.py
```

### 3. Launch Flask Cybersecurity Web Application
```bash
python flask_app.py
```
Open **`http://localhost:5000`** in your browser to access:
- **Analyze SMS:** Test messages with character count, presets, real model inference, and explainable indicators.
- **Dashboard:** Live session telemetry, risk distribution, and historical logs.
- **Model Performance:** Table 1 from the paper with dynamic metrics, confusion matrix, and error analysis.
- **How It Works:** Visual 7-step pipeline from raw input to explainable output.
- **Dataset:** Specification of UCI corpus (5,574 instances) and boundaries.
- **Research & Viva:** Synopsis, objectives, methodology, limitations, and ethical considerations.

### 4. Or Run Streamlit Dashboard
```bash
streamlit run app.py
```
Accessible at **`http://localhost:8501`**.

### 5. Run Automated CLI Test Suite
```bash
python test_system.py
```

---

## 🔒 Privacy, Safety & Ethical Disclaimer
> **Important Notice:** This tool provides an AI-assisted risk assessment. It does not verify the sender’s identity or guarantee that a message is fraudulent or legitimate. These caution flags are warning indicators only and must never be described as absolute proof of fraud. All message data is processed locally; no contents are transmitted to external services. Users should not input sensitive private credentials into demonstration applications.
