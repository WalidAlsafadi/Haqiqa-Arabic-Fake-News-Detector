import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.text_cleaner import clean_arabic_text, clean_arabic_text_minimal

# -------------------------------------
# Optional outlier removal (IQR method)
# -------------------------------------
def remove_outliers_iqr(df: pd.DataFrame, col: str = 'content_length') -> pd.DataFrame:
    """Remove outliers based on IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

# ----------------------------
# Configuration
# ----------------------------
CLEANED_PATH = "data/processed/cleaned_news.csv"
REMOVE_OUTLIERS = True

# ----------------------------
# Load dataset
# ----------------------------
print("🔹 Loading cleaned dataset...")
df = pd.read_csv(CLEANED_PATH)

# ----------------------------
# Text Cleaning: Dual Modes
# ----------------------------
print("🔹 Creating dual-cleaned text columns...")

# Aggressive clean (stopwords + stemming)
df["title_clean_agg"] = df["title"].astype(str).apply(clean_arabic_text)
df["content_clean_agg"] = df["content"].astype(str).apply(clean_arabic_text)
df["text_clean_agg"] = df["title_clean_agg"] + " " + df["content_clean_agg"]

# Minimal clean (no stopwords, no stemming)
df["title_clean_min"] = df["title"].astype(str).apply(clean_arabic_text_minimal)
df["content_clean_min"] = df["content"].astype(str).apply(clean_arabic_text_minimal)
df["text_clean_min"] = df["title_clean_min"] + " " + df["content_clean_min"]

# ➕ Add content-only text fields
df["text_clean_agg_content_only"] = df["content_clean_agg"]
df["text_clean_min_content_only"] = df["content_clean_min"]

# ----------------------------
# Feature Engineering
# ----------------------------
df["title_length"] = df["title"].apply(lambda x: len(str(x).split()))
df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))

if REMOVE_OUTLIERS:
    print("🔹 Removing outliers...")
    df = remove_outliers_iqr(df, col="content_length")

# ----------------------------
# Label Encoding
# ----------------------------
df["label"] = df["label"].map({"real": 0, "fake": 1})
df["platform_encoded"] = LabelEncoder().fit_transform(df["platform_grouped"])

# ----------------------------
# Save Outputs
# ----------------------------
print("🔹 Saving modeling datasets...")

# Text-only (aggressive and minimal)
df[["text_clean_agg", "label"]].to_csv("data/processed/text_clean_agg.csv", index=False)
df[["text_clean_min", "label"]].to_csv("data/processed/text_clean_min.csv", index=False)

# Save new text-only content version
df_content_only = df[["text_clean_min_content_only", "label"]].rename(columns={"text_clean_min_content_only": "text"})
df_content_only.to_csv("data/processed/text_clean_min_content_only.csv", index=False)


# Full feature set for classical models
df_full = df[[
    "text_clean_agg", "title_length", "content_length", "platform_encoded", "label"
]]
df_full.to_csv("data/processed/with_features.csv", index=False)

# ----------------------------
# Summary
# ----------------------------
print("✅ Preprocessing complete.")
print(f"🗂 Records: {len(df)}")
print("📊 Class distribution:")
print(df['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
