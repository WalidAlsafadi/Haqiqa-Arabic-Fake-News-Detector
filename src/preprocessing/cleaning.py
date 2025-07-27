import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from arabicstopwords.arabicstopwords import stopwords_list
from src.utils.logger import log_info, log_step, log_success

import nltk
nltk.download("punkt")

ARABIC_STOPWORDS = set(stopwords_list())

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    return text

def remove_diacritics(text):
    return re.sub(r'[\u064B-\u0652]', '', text)

def remove_non_arabic(text):
    return re.sub(r"[^\u0600-\u06FF\s]", " ", text)

def clean_arabic_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def clean_arabic_text_minimal(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_outliers_iqr(df: pd.DataFrame, col: str = 'content_length') -> pd.DataFrame:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

def prepare_text_for_transformers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"News content": "content", "Label": "label"})

    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["text"] = df["title"] + " " + df["content"]
    df["label"] = df["label"].str.lower().map({"real": 0, "fake": 1})

    df = df[["text", "label"]].dropna()
    log_success("Transformer-ready dataset prepared.")
    return df

def apply_all_cleaning(path: str, remove_outliers: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    log_step("Standardizing columns and parsing dates...")
    df = df.rename(columns={"Id": "id", "News content": "content", "Label": "label"})
    df['date'] = pd.to_datetime(df['date'])
    df['label'] = df['label'].str.lower()

    log_step("Removing duplicates and filtering by date...")
    df = df.drop_duplicates(subset=['content'])
    df = df.drop_duplicates(subset=['title'])
    df = df[df['date'] >= '2023-01-01']

    log_step("Grouping low-frequency platforms...")
    df['platform_grouped'] = df['platform'].apply(
        lambda x: x if df['platform'].value_counts()[x] >= 30 else 'Other')
    df = df.drop(columns=['id', 'platform'])

    log_step("Applying aggressive Arabic text cleaning...")
    df["title_clean_agg"] = df["title"].astype(str).apply(clean_arabic_text)
    df["content_clean_agg"] = df["content"].astype(str).apply(clean_arabic_text)
    df["text_agg"] = df["title_clean_agg"] + " " + df["content_clean_agg"]

    log_step("Applying minimal Arabic text cleaning...")
    df["title_clean_min"] = df["title"].astype(str).apply(clean_arabic_text_minimal)
    df["content_clean_min"] = df["content"].astype(str).apply(clean_arabic_text_minimal)
    df["text_min"] = df["title_clean_min"] + " " + df["content_clean_min"]

    log_step("Generating feature columns...")
    df["title_length"] = df["title"].apply(lambda x: len(str(x).split()))
    df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))

    if remove_outliers:
        log_step("Removing outliers using IQR method...")
        original_len = len(df)
        df = remove_outliers_iqr(df, col="content_length")
        log_info(f"Removed {original_len - len(df)} outliers")

    log_step("Encoding labels and platforms...")
    df['label'] = df['label'].map({'real': 0, 'fake': 1})
    df['platform_encoded'] = LabelEncoder().fit_transform(df['platform_grouped'])

    log_success("Data cleaning completed.")
    return df
