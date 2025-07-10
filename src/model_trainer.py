import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from src.features import build_tfidf_vectorizer, transform_texts, save_vectorizer
import joblib
import os

# -------------------------------
# Configs
# -------------------------------
DATA_PATH = "data/processed/text_clean_agg.csv"   # Try text_clean_min.csv for variation
USE_HYBRID = False  # If True, use with_features.csv
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load Data
# -------------------------------
print(f"🔹 Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X_text = df["text_clean_agg"]
y = df["label"]

# -------------------------------
# Vectorize
# -------------------------------
vectorizer, X_vec = build_tfidf_vectorizer(X_text)
save_vectorizer(vectorizer, path=os.path.join(MODEL_OUTPUT_DIR, "tfidf_vectorizer.pkl"))

# Optional hybrid features
if USE_HYBRID:
    df_features = pd.read_csv("data/processed/with_features.csv")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_features[["title_length", "content_length", "platform_encoded"]])
    X_final = hstack([X_vec, X_num])
    joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, "feature_scaler.pkl"))
else:
    X_final = X_vec

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Model Candidates
# -------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

results = []

# -------------------------------
# Train + Evaluate
# -------------------------------
for name, model in models.items():
    print(f"\n🚀 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, f"{name}.pkl"))
    results.append({"Model": name, "Accuracy": acc, "F1": f1})

# -------------------------------
# Save Comparison Table
# -------------------------------
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv("models/model_comparison.csv", index=False)
print("\n📊 Saved model comparison to models/model_comparison.csv")
