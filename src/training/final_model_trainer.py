import os
import sys
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.config import (
    DATA_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    SEED,
    TEXT_COLUMN_NAME
)
from src.utils.logger import log_info, log_step, log_success

# -------------------------------
# Choose Model Type
# -------------------------------
MODEL_TYPE = "xgb"  # or "rf"

# -------------------------------
# Setup
# -------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load & Split Data
# -------------------------------
log_step(f"Loading cleaned dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df[TEXT_COLUMN_NAME]
y = df["label"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=SEED)

X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])

# -------------------------------
# Build Pipeline
# -------------------------------
log_step(f"Building TF-IDF + {MODEL_TYPE.upper()} pipeline...")

if MODEL_TYPE == "xgb":
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=1000,
        eval_metric="aucpr",
        scale_pos_weight=1.8,
        random_state=SEED
    )
    model_filename = "model_final_xgb_pipeline.pkl"

elif MODEL_TYPE == "rf":
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED
    )
    model_filename = "model_final_rf_pipeline.pkl"

else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", clf)
])

# -------------------------------
# Training
# -------------------------------
log_step("Training pipeline on full train+val set...")
start = time.time()
pipeline.fit(X_final, y_final)
duration = time.time() - start
log_info(f"Training time: {duration:.2f} seconds")

# Save full pipeline
joblib.dump(pipeline, os.path.join(MODEL_DIR, model_filename))
log_success(f"Pipeline saved to {model_filename}")

# -------------------------------
# Cross-Validation
# -------------------------------
log_step("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(pipeline, X_final, y_final, cv=5, scoring="f1")
log_success(f"CV F1 Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

# -------------------------------
# Final Evaluation on Test
# -------------------------------
log_step("Evaluating on test set...")
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print(report)
log_success(f"Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")

# Save predictions
eval_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
eval_df.to_csv(os.path.join(OUTPUT_DIR, "test_eval.csv"), index=False)

# Save text report
with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

log_success("Test evaluation results saved.")
