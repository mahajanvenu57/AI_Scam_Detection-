# SYMCA Semester III Research Project Report
# SCAMSHIELD AI: AI-Based Scam Detection & Risk Analysis (MCARP31)

**Prepared by:**
- Harshi Anand Naik (Roll No: 24)
- Venu Mahajan (Roll No: 15)
- Sayali Shelake (Roll No: 41)

---

## 1. Executive Summary & Experimental Methodology
This research establishes an **Explainable Artificial Intelligence (XAI)** and supervised machine learning system for SMS phishing and fraud categorization. The evaluation was executed on the official **UCI SMS Spam Collection dataset (5,574 instances)** under an **80/20 stratified split** with fixed random seed ($42$) and stratified 5-fold cross-validation.

To avoid data leakage, text preprocessing and sublinear **TF-IDF vectorization (unigrams + bigrams across 5,000 dimensions)** were fitted strictly on the training partition ($N = 4,459$). The held-out test partition ($N = 1,115$) was transformed strictly through the frozen training vocabulary.

---

## 2. Populated Benchmark Table (Section 8.3 of Paper)

The table below replaces the *"Not calculated"* entries from the synopsis submission with genuine, empirically calculated performance metrics:

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | ROC-AUC | 5-Fold CV F1 | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** ⭐ | **0.9839** | **0.9712** | **0.9060** | **0.9375** | **0.9915** | **0.9428** | **0.0041** | **0.0940** |
| **Linear Support Vector Machine** | **0.9839** | **0.9712** | **0.9060** | **0.9375** | **0.9914** | **0.9326** | **0.0041** | **0.0940** |
| **Random Forest** | 0.9776 | 1.0000 | 0.8322 | 0.9084 | 0.9877 | 0.9161 | 0.0000 | 0.1678 |
| **Logistic Regression** | 0.9767 | 1.0000 | 0.8255 | 0.9044 | 0.9891 | 0.7743 | 0.0000 | 0.1745 |
| **Decision Tree** | 0.9525 | 0.8380 | 0.7987 | 0.8179 | 0.8631 | 0.8486 | 0.0238 | 0.2013 |

---

## 3. Confusion Matrix Breakdown (Test Partition: N = 1,115)

```
Multinomial Naive Bayes (Baseline Winner):
-------------------------------------------------------
                           Predicted Ham   Predicted Spam
Actual Ham (966 msgs):         962 (TN)            4 (FP)
Actual Spam (149 msgs):         14 (FN)          135 (TP)
-------------------------------------------------------
Total Accuracy: 98.39% | F1: 0.9375 | ROC-AUC: 0.9915
```

- **True Negatives (TN = 962):** Legitimate personal or transactional messages correctly identified as Ham.
- **False Positives (FP = 4):** Legitimate messages mistakenly flagged as Spam (0.41% FPR).
- **False Negatives (FN = 14):** Scam messages that eluded TF-IDF keyword classification (9.40% FNR).
- **True Positives (TP = 135):** Malicious communications and lottery smishing correctly caught.

---

## 4. In-Depth Comparative Discussion

1. **Why Multinomial Naive Bayes & Linear SVM Dominated:**
   - Text vectorization produces high-dimensional sparse representations. Naive Bayes handles feature conditional independence exceptionally well on sparse text, while Linear SVM finds the optimal hyper-plane margin without suffering from the curse of dimensionality.
2. **The Precision vs. Recall Security Dilemma:**
   - Both **Logistic Regression** and **Random Forest** achieved **1.0000 (100%) Precision**, resulting in zero false alarms ($FP = 0$).
   - However, their Recall was restricted to 82.55% and 83.22%, allowing 17% of scams through. In security systems, an undetected scam ($FN$) creates a direct attack surface for financial fraud. Naive Bayes achieves the ideal operational equilibrium by reducing missed scams to only 9.4%.

---

## 5. Explainable AI (XAI) & 10 Caution Indicators

SCAMSHIELD AI bridges black-box ML inference with an explainable multi-point heuristic analyzer:
1. **Urgent Language:** Flags artificial deadlines (`urgent`, `immediately`, `within 24 hours`).
2. **Pressure / Threat Language:** Flags coercive threats (`suspended`, `blocked`, `legal action`).
3. **Prize / Reward Language:** Flags unsolicited winnings (`won`, `claim prize`, `lucky draw`).
4. **Financial Request:** Flags monetary triggers (`bank`, `account`, `pre-approved loan`).
5. **Credential / Password Request:** Identifies sensitive harvesting (`KYC`, `OTP`, `PIN`, `password`).
6. **Suspicious URLs:** Regex extraction of unverified web domains and smishing links.
7. **Currency Symbols:** Identifies monetary quantifiers ($, GBP, EUR, INR, Rs).
8. **Phone / Shortcodes:** Identifies premium-rate 5-digit shortcodes and unverified numbers.
9. **Excessive Numbers:** Flags high numeric density typical of fake serial numbers and transaction IDs.
10. **Suspicious Keywords & Shouting:** Flags excessive capitalization (>25%) and spam directives.

---

## 6. Privacy & Safety Notice
*This tool provides an AI-assisted risk assessment. It does not verify the sender’s identity or guarantee that a message is fraudulent or legitimate. These caution flags are warning indicators only and must never be described as absolute proof of fraud. All analysis occurs locally without transmitting text to external services.*
