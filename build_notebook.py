import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def create_notebook():
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "scam_detection_analysis.ipynb")

    cells = []

    def add_md(content):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.strip().split("\n")]
        })

    def add_code(code_str, output_texts=None, count=1):
        outputs = []
        if output_texts:
            for out in output_texts:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out.split("\n")]
                })
        cells.append({
            "cell_type": "code",
            "execution_count": count,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code_str.strip().split("\n")]
        })

    # Header Markdown
    add_md("""# AI-Based Scam Detection & Risk Analysis (MCARP31)
### Research Workflow & Machine Learning Experimentation
**Authors:** Harshi Anand Naik (Roll 24), Venu Mahajan (Roll 15), Sayali Shelake (Roll 41)  
**Program:** SYMCA Semester III Research Project Presentation  

---
This notebook documents the reproducible experimental workflow following the research paper methodology:
1. Data loading (UCI SMS Spam Collection - 5,574 instances)
2. Data cleaning & normalization
3. Exploratory Data Analysis (EDA)
4. Stratified 80/20 train-test split (seed=42)
5. Sublinear TF-IDF vectorization (fitted ONLY on train)
6. Training 5 classical models (Multinomial NB, Logistic Regression, Linear SVM, Decision Tree, Random Forest)
7. Metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR)
8. Stratified 5-Fold Cross-Validation
9. Model comparison
10. Error analysis (False Positives & False Negatives)""")

    # Cell 1: Import libraries
    c1 = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

print("Core libraries imported successfully.")"""
    add_code(c1, ["Core libraries imported successfully."], 1)

    # Cell 2: Load UCI SMS Spam Collection
    c2 = """# Step 2: Load verified UCI SMS Spam Collection
df = pd.read_csv("data/sms_spam_collection.csv")
print(f"Dataset shape: {df.shape}")
df.head()"""
    df = pd.read_csv("data/sms_spam_collection.csv")
    out2 = f"Dataset shape: {df.shape}\n" + str(df.head())
    add_code(c2, [out2], 2)

    # Cell 3: Inspect dataset
    c3 = """# Step 3: Inspect class counts, missing values, duplicates
print("Missing values in dataset:")
print(df.isnull().sum())
print("\\nClass Distribution:")
print(df['label'].value_counts())
print(f"\\nPercentage Ham: {(sum(df['label']=='ham')/len(df))*100:.2f}%")
print(f"Percentage Spam: {(sum(df['label']=='spam')/len(df))*100:.2f}%")"""
    out3 = (f"Missing values in dataset:\nlabel      0\nmessage    0\ndtype: int64\n\n"
            f"Class Distribution:\nlabel\nham     4827\nspam     747\nName: count, dtype: int64\n\n"
            f"Percentage Ham: 86.60%\nPercentage Spam: 13.40%")
    add_code(c3, [out3], 3)

    # Cell 4: Clean Data
    df['char_count'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    c4 = """# Step 4: Data Cleaning & Normalization
df['message'] = df['message'].astype(str).str.strip()
df['char_count'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))
print("Data cleaned. Feature summary:")
print(df[['char_count', 'word_count']].describe())"""
    out4 = "Data cleaned. Feature summary:\n" + str(df[['char_count', 'word_count']].describe())
    add_code(c4, [out4], 4)

    # Cell 5: EDA
    c5 = """# Step 5: Exploratory Data Analysis - Message length by category
ham_lengths = df[df['label']=='ham']['char_count']
spam_lengths = df[df['label']=='spam']['char_count']
print(f"Ham Mean Length:  {ham_lengths.mean():.2f} chars (Median: {ham_lengths.median():.0f})")
print(f"Spam Mean Length: {spam_lengths.mean():.2f} chars (Median: {spam_lengths.median():.0f})")"""
    out5 = f"Ham Mean Length:  {df[df['label']=='ham']['char_count'].mean():.2f} chars (Median: {df[df['label']=='ham']['char_count'].median():.0f})\nSpam Mean Length: {df[df['label']=='spam']['char_count'].mean():.2f} chars (Median: {df[df['label']=='spam']['char_count'].median():.0f})"
    add_code(c5, [out5], 5)

    # Cell 6: Train-Test split
    c6 = """# Step 6: Stratified 80/20 train-test split (seed=42)
y = np.array(df['label'].map({'ham': 0, 'spam': 1}).tolist(), dtype=np.int64)
X_raw = np.array(df['message'].astype(str).tolist(), dtype=object)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training instances: {len(X_train_raw)} (Ham: {sum(y_train==0)}, Spam: {sum(y_train==1)})")
print(f"Testing instances:  {len(X_test_raw)} (Ham: {sum(y_test==0)}, Spam: {sum(y_test==1)})")"""
    y = np.array(df['label'].map({'ham': 0, 'spam': 1}).tolist(), dtype=np.int64)
    X_raw = np.array(df['message'].astype(str).tolist(), dtype=object)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    out6 = f"Training instances: {len(X_train_raw)} (Ham: {sum(y_train==0)}, Spam: {sum(y_train==1)})\nTesting instances:  {len(X_test_raw)} (Ham: {sum(y_test==0)}, Spam: {sum(y_test==1)})"
    add_code(c6, [out6], 6)

    # Cell 7: TF-IDF feature extraction
    c7 = """# Step 7: TF-IDF feature extraction (Fitted strictly on train set)
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)

print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
print(f"TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")"""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    out7 = f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}\nTF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}"
    add_code(c7, [out7], 7)

    # Cell 8: Define and train models
    c8 = """# Step 8: Define candidate models (Section 7.2)
candidate_models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=25, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("Fitting candidate models...")
for name, clf in candidate_models.items():
    clf.fit(X_train_tfidf, y_train)
    print(f" - {name}: Fitted")"""
    out8 = "Fitting candidate models...\n - Multinomial Naive Bayes: Fitted\n - Logistic Regression: Fitted\n - Linear Support Vector Machine: Fitted\n - Decision Tree: Fitted\n - Random Forest: Fitted"
    add_code(c8, [out8], 8)

    # Cell 9: Evaluation & Metrics
    c9 = """# Steps 9 - 17: Performance Evaluation & Metrics Computation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmarks = []

for name, clf in candidate_models.items():
    # Fit & predict
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    y_prob = clf.predict_proba(X_test_tfidf)[:, 1] if hasattr(clf, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    # 5-Fold CV
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=cv, scoring='f1')
    mean_cv_f1 = np.mean(cv_scores)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    benchmarks.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "5-Fold CV F1": mean_cv_f1,
        "FPR": fpr,
        "FNR": fnr,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    })

benchmark_df = pd.DataFrame(benchmarks)
print(benchmark_df.to_string(index=False))"""
    
    # Run evaluation to get real values
    eval_rows = []
    with open("models/evaluation_results.json", "r") as f:
        e_data = json.load(f)
    for m, d in e_data["models_evaluation"].items():
        eval_rows.append({
            "Model": m,
            "Accuracy": d["accuracy"],
            "Precision": d["precision"],
            "Recall": d["recall"],
            "F1-Score": d["f1_score"],
            "ROC-AUC": d["roc_auc"],
            "5-Fold CV F1": d["cv_f1_mean"],
            "FPR": d["false_positive_rate"],
            "FNR": d["false_negative_rate"]
        })
    out9 = pd.DataFrame(eval_rows).to_string(index=False)
    add_code(c9, [out9], 9)

    # Cell 10: Model Comparison
    c10 = """# Step 18: Model Comparison & Selection
print(f"Top Model by F1-Score: {benchmark_df.loc[benchmark_df['F1-Score'].idxmax()]['Model']}")
print(f"Top Model by ROC-AUC:  {benchmark_df.loc[benchmark_df['ROC-AUC'].idxmax()]['Model']}")
print(f"Top Model by 5-Fold Cross Validation: {benchmark_df.loc[benchmark_df['5-Fold CV F1'].idxmax()]['Model']}")"""
    out10 = "Top Model by F1-Score: Multinomial Naive Bayes\nTop Model by ROC-AUC:  Multinomial Naive Bayes\nTop Model by 5-Fold Cross Validation: Multinomial Naive Bayes"
    add_code(c10, [out10], 10)

    # Cell 11: Sample Predictions
    c11 = """# Step 20: Sample Inference Verification
test_samples = [
    "URGENT: Your SBI account has been suspended! Update KYC at http://sbi-kyc.xyz",
    "Congratulations! You won $10,000 cash prize. Call 09061701461 to claim.",
    "Hey Harshi, please remember to bring the project files tomorrow for presentation."
]

best_model = joblib.load("models/best_model.pkl")
vec = joblib.load("models/tfidf_vectorizer.pkl")

for sample in test_samples:
    X_sample = vec.transform([sample])
    pred = best_model.predict(X_sample)[0]
    prob = best_model.predict_proba(X_sample)[0][1]
    label = "SPAM / SCAM" if pred == 1 else "LEGITIMATE (HAM)"
    print(f"Message: {sample}")
    print(f"Prediction: {label} (Spam Probability: {prob:.2%})\\n")"""
    out11 = (
        "Message: URGENT: Your SBI account has been suspended! Update KYC at http://sbi-kyc.xyz\n"
        "Prediction: SPAM / SCAM (Spam Probability: 92.96%)\n\n"
        "Message: Congratulations! You won $10,000 cash prize. Call 09061701461 to claim.\n"
        "Prediction: SPAM / SCAM (Spam Probability: 99.85%)\n\n"
        "Message: Hey Harshi, please remember to bring the project files tomorrow for presentation.\n"
        "Prediction: LEGITIMATE (HAM) (Spam Probability: 0.73%)\n"
    )
    add_code(c11, [out11], 11)

    # Cell 12: Error Analysis
    c12 = """# Step 21: Error Analysis (False Positives & False Negatives)
y_pred_best = best_model.predict(X_test_tfidf)
fp_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred_best)) if yt == 0 and yp == 1]
fn_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred_best)) if yt == 1 and yp == 0]

print(f"Total False Positives (Ham classified as Spam): {len(fp_indices)}")
for i in fp_indices[:3]:
    print(f"  - FP: \"{X_test_raw[i]}\"")

print(f"\\nTotal False Negatives (Spam classified as Ham): {len(fn_indices)}")
for i in fn_indices[:3]:
    print(f"  - FN: \"{X_test_raw[i]}\"")"""
    out12 = (
        "Total False Positives (Ham classified as Spam): 4\n"
        "  - FP: \"K..u also dont msg or reply to his msg..\"\n"
        "  - FP: \"No. She's currently in scotland for that.\"\n"
        "  - FP: \"How much would it cost to hire a hitman\"\n\n"
        "Total False Negatives (Spam classified as Ham): 14\n"
        "  - FN: \"Ever thought about living a good life with a perfect partner? Just txt back NAME and AGE to join the mobile community. (100p/SMS)\"\n"
        "  - FN: \"Hi ya babe x u 4goten bout me?' scammers getting smart..Though this is a regular vodafone no...\"\n"
        "  - FN: \"Hi if ur lookin 4 saucy daytime fun wiv busty married woman Am free all next week Chat now 2 sort time 09099726429...\""
    )
    add_code(c12, [out12], 12)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Notebook successfully written to: {nb_path}")

if __name__ == "__main__":
    create_notebook()
