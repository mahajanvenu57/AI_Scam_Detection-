import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

from dataset import ensure_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

def run_training_pipeline():
    print("=" * 60)
    print("AI-BASED SCAM DETECTION - TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load dataset
    print("[1/6] Loading verified SMS Spam Collection dataset...")
    df = ensure_dataset()
    print(f"Total rows: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    # EDA features
    df['char_count'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))

    # Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', hue='label', palette=['#3b82f6', '#ef4444'], legend=False)
    plt.title("Class Distribution (Ham vs Spam)")
    plt.xlabel("Category")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "class_distribution.png"), dpi=200)
    plt.close()

    # Plot 2: Message Length Distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x='char_count', hue='label', bins=50, kde=True, palette=['#3b82f6', '#ef4444'])
    plt.title("Character Count Distribution: Ham vs Spam")
    plt.xlabel("Character Length")
    plt.ylabel("Frequency")
    plt.xlim(0, 300)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "length_distribution.png"), dpi=200)
    plt.close()

    # 2. Encode labels: ham -> 0, spam -> 1
    y = np.array(df['label'].map({'ham': 0, 'spam': 1}).tolist(), dtype=np.int64)
    X_raw = np.array(df['message'].astype(str).tolist(), dtype=object)

    # 3. Stratified train-test split (80-20 split, seed=42 for reproducibility)
    print("\n[2/6] Performing Stratified Train-Test Split (80/20, seed=42)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train_raw)} | Test samples: {len(X_test_raw)}")

    # 4. TF-IDF Vectorization (fit ONLY on training data to prevent data leakage)
    print("\n[3/6] Fitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

    # Save Vectorizer
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    # 5. Define candidate models matching Section 7.2 of paper
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=25, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # 6. Train and Evaluate each model
    print("\n[4/6] Training and evaluating candidate models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_f1 = -1
    best_model_name = None

    for name, clf in models.items():
        print(f"--> Training: {name}...")
        clf.fit(X_train_tfidf, y_train)

        # 5-fold cross validation score on training set
        cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=cv, scoring='f1')
        mean_cv_f1 = float(np.mean(cv_scores))

        # Test set predictions
        y_pred = clf.predict(X_test_tfidf)
        y_proba = clf.predict_proba(X_test_tfidf)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "cv_f1_mean": round(mean_cv_f1, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "classification_report": classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
        }

        # Save individual model
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(clf, os.path.join(MODELS_DIR, filename))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # Save best model reference as default_model.pkl
    best_clf = models[best_model_name]
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"\n[5/6] Best Model identified: {best_model_name} (F1 Score: {best_f1:.4f})")

    # 7. Generate Visualizations
    print("[6/6] Generating evaluation visual artifacts...")

    # Bar chart comparing models
    model_names_list = list(results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    plot_df = []
    for m in model_names_list:
        for metric in metrics_to_plot:
            plot_df.append({
                'Model': m,
                'Metric': metric.replace('_', ' ').capitalize(),
                'Score': results[m][metric]
            })
    plot_df = pd.DataFrame(plot_df)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title("Comparative Performance across Candidate Models", fontsize=14, fontweight='bold')
    plt.ylim(0.8, 1.02)
    plt.xticks(rotation=15, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "model_comparison.png"), dpi=200)
    plt.close()

    # Confusion matrix of best model
    plt.figure(figsize=(5, 4))
    best_cm = np.array([
        [results[best_model_name]["confusion_matrix"]["tn"], results[best_model_name]["confusion_matrix"]["fp"]],
        [results[best_model_name]["confusion_matrix"]["fn"], results[best_model_name]["confusion_matrix"]["tp"]]
    ])
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Ham', 'Pred Spam'],
                yticklabels=['Actual Ham', 'Actual Spam'])
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "best_confusion_matrix.png"), dpi=200)
    plt.close()

    # Extract False Positives and False Negatives for Error Analysis (Section 8.2 & 8.3)
    best_preds = best_clf.predict(X_test_tfidf)
    error_analysis = {
        "false_positives": [], # Ham messages predicted as spam
        "false_negatives": []  # Spam messages predicted as ham
    }

    for text, actual, pred in zip(X_test_raw, y_test, best_preds):
        if actual == 0 and pred == 1:
            error_analysis["false_positives"].append(text)
        elif actual == 1 and pred == 0:
            error_analysis["false_negatives"].append(text)

    # Save summary json
    summary = {
        "dataset_total": len(df),
        "ham_count": int(sum(df['label'] == 'ham')),
        "spam_count": int(sum(df['label'] == 'spam')),
        "train_count": len(X_train_raw),
        "test_count": len(X_test_raw),
        "best_model": best_model_name,
        "models_evaluation": results,
        "error_analysis_summary": {
            "total_false_positives": len(error_analysis["false_positives"]),
            "total_false_negatives": len(error_analysis["false_negatives"]),
            "sample_false_positives": error_analysis["false_positives"][:5],
            "sample_false_negatives": error_analysis["false_negatives"][:5]
        }
    }

    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Results saved to {os.path.join(MODELS_DIR, 'evaluation_results.json')}")
    print("\nTraining and Evaluation complete successfully!")
    return summary

if __name__ == "__main__":
    run_training_pipeline()
