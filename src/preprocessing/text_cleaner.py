"""Arabic text cleaning utilities with minimal and aggressive approaches"""

import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from arabicstopwords.arabicstopwords import stopwords_list
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

# Arabic stopwords from proper library
ARABIC_STOPWORDS = set(stopwords_list())

def normalize_arabic(text):
    """Normalize Arabic characters"""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    return text

def remove_diacritics(text):
    """Remove Arabic diacritics"""
    return re.sub(r'[\u064B-\u0652]', '', text)

def remove_non_arabic(text):
    """Keep only Arabic characters and spaces"""
    return re.sub(r"[^\u0600-\u06FF\s]", " ", text)

def clean_arabic_text_aggressive(text):
    """Aggressive Arabic text cleaning with stopword removal"""
    if pd.isna(text): 
        return ""
    
    text = text.lower()
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
    
    return " ".join(tokens)

def clean_arabic_text_transformers(text):
    """Minimal cleaning for transformers (AraBERT) - preserve original structure"""
    if pd.isna(text): 
        return ""
    
    # Very light cleaning - keep most structure for transformers
    text = re.sub(r"\s+", " ", text).strip()  # Only normalize whitespace
    text = text[:512]  # Limit for transformer models
    
    return text

def clean_arabic_text_minimal(text):
    """Minimal Arabic text cleaning - keeps more context"""
    if pd.isna(text): 
        return ""
    
    text = text.lower()
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def remove_outliers_iqr(df: pd.DataFrame, col: str = 'content_length') -> pd.DataFrame:
    """Remove outliers using IQR method"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

def prepare_data(df):
    """Complete data preparation with both cleaning approaches"""
    
    print("Standardizing columns and parsing dates...")
    # Rename columns for consistency
    if 'News content' in df.columns:
        df = df.rename(columns={"Id": "id", "News content": "content", "Label": "label"})
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'label' in df.columns:
        df['label'] = df['label'].str.lower()

    print("Removing duplicates and filtering...")
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['content'])
    df = df.drop_duplicates(subset=['title'])
    print(f"Removed {original_len - len(df)} duplicates")
    
    # Filter by date if available
    if 'date' in df.columns:
        df = df[df['date'] >= '2023-01-01']
    
    # Group low-frequency platforms if platform column exists
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        df['platform_grouped'] = df['platform'].apply(
            lambda x: x if platform_counts[x] >= 30 else 'Other'
        )

    print("Applying aggressive Arabic text cleaning...")
    df["title_clean_agg"] = df["title"].astype(str).apply(clean_arabic_text_aggressive)
    df["content_clean_agg"] = df["content"].astype(str).apply(clean_arabic_text_aggressive)
    df["text_aggressive"] = df["title_clean_agg"] + " " + df["content_clean_agg"]

    print("Applying minimal Arabic text cleaning...")
    df["title_clean_min"] = df["title"].astype(str).apply(clean_arabic_text_minimal)
    df["content_clean_min"] = df["content"].astype(str).apply(clean_arabic_text_minimal)
    df["text_minimal"] = df["title_clean_min"] + " " + df["content_clean_min"]

    print("Applying transformers-ready text cleaning...")
    df["title_clean_transformers"] = df["title"].astype(str).apply(clean_arabic_text_transformers)
    df["content_clean_transformers"] = df["content"].astype(str).apply(clean_arabic_text_transformers)
    df["text_transformers"] = df["title_clean_transformers"] + " " + df["content_clean_transformers"]

    print("Generating feature columns...")
    df["title_length"] = df["title"].apply(lambda x: len(str(x).split()))
    df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))

    # Remove outliers
    print("Removing outliers using IQR method...")
    original_len = len(df)
    df = remove_outliers_iqr(df, col="content_length")
    print(f"Removed {original_len - len(df)} outliers")

    # Encode labels and platforms
    if 'label' in df.columns:
        df['label'] = df['label'].map({'real': 0, 'fake': 1})
    if 'platform_grouped' in df.columns:
        df['platform_encoded'] = LabelEncoder().fit_transform(df['platform_grouped'])

    # Remove very short content
    df = df[df['content_length'] > 10].copy()
    
    print(f"Data cleaning completed - {len(df)} articles ready")
    return df
