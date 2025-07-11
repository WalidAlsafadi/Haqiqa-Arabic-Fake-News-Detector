import os
import sys
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Adjust import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.config import DATA_PATH, FEATURE_PATH, MODEL_DIR, OUTPUT_DIR, USE_HYBRID, SEED
from src.utils.logger import log_info, log_success, log_step, log_fail

# -------------------------
# Ensure output directories
# -------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Dataset
# -------------------------
log_step(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X_text = df["text_min"]  # or text_agg depending on config
y = df["label"]

# -------------------------
# Train / Val / Test Split
# -------------------------
X_temp, X_test, y_temp, y_test = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=SEED)

# Save test set for later
joblib.dump((X_test, y_test), os.path.join(OUTPUT_DIR, "test_eval.joblib"))

# -------------------------
# Model Candidates (as Pipelines)
# -------------------------
results = []

pipelines = {
    "LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    ]),
    "MultinomialNB": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ]),
    "RandomForest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED))
    ]),
    "XGBoost": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", XGBClassifier(n_estimators=100, eval_metric="logloss", scale_pos_weight=1.8, random_state=SEED))
    ])
}

# -------------------------
# Training + Evaluation Loop
# -------------------------
for name, pipeline in pipelines.items():
    if USE_HYBRID and name == "MultinomialNB":
        log_info(f"Skipping {name} — incompatible with scaled features.")
        continue

    log_step(f"Training: {name}")
    try:
        start = time.time()
        pipeline.fit(X_train, y_train)
        duration = time.time() - start

        y_val_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        report = classification_report(y_val, y_val_pred, digits=3)
        print(report)

        log_success(f"{name} — Val Accuracy: {acc:.4f}, F1: {f1:.4f}, Time: {duration:.2f}s")

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

        # Save results
        results.append({
            "Model": name,
            "Val_Accuracy": acc,
            "Val_F1": f1,
            "CV_F1_Mean": cv_scores.mean(),
            "CV_F1_Std": cv_scores.std()
        })

        # Save full pipeline
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))

    except Exception as e:
        log_fail(f"{name} failed: {str(e)}")

# -------------------------
# Save Evaluation Summary
# -------------------------
results_df = pd.DataFrame(results).sort_values("Val_F1", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_eval_val_min.csv"), index=False)
log_success("Validation results saved to model_eval_val_min.csv")
