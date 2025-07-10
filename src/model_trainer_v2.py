import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.sparse import hstack
from src.features import build_tfidf_vectorizer, save_vectorizer
import joblib

# -------------------------------
# Configs
# -------------------------------
USE_HYBRID = False
DATA_PATH = "data/processed/text_clean_min_content_only.csv"  # Or text_clean_min.csv
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
print(f"🔹 Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X_text = df["text"]
y = df["label"]

# -------------------------------
# Vectorization
# -------------------------------
vectorizer, X_vec = build_tfidf_vectorizer(X_text)
save_vectorizer(vectorizer, os.path.join(MODEL_OUTPUT_DIR, "tfidf_vectorizer.pkl"))

# Optional: Hybrid input
if USE_HYBRID:
    df_feat = pd.read_csv("data/processed/with_features.csv")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_feat[["title_length", "content_length", "platform_encoded"]])
    joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, "feature_scaler.pkl"))
    X_final = hstack([X_vec, X_num])
else:
    X_final = X_vec

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# Model Candidates
# -------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "MultinomialNB": MultinomialNB(),  # NB doesn’t use class_weight
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=100, eval_metric="logloss", scale_pos_weight=1.8
    )
}

results = []

# -------------------------------
# Training Loop
# -------------------------------
for name, model in models.items():
    if USE_HYBRID and name == "MultinomialNB":
        print(f"⚠️ Skipping {name} — not compatible with negative values from StandardScaler.")
        continue

    print(f"\n🚀 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    cv_f1_mean = cv_scores.mean()
    cv_f1_std = cv_scores.std()

    print(f"📊 CV F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    # Save model
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)

    # Log results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1": f1,
        "CV_F1_Mean": cv_f1_mean,
        "CV_F1_Std": cv_f1_std
    })

# -------------------------------
# Save Summary Table
# -------------------------------
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "model_comparison_v2.csv"), index=False)
print("\n✅ Model training complete. Results saved to model_comparison_v2.csv")
