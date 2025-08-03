import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from arabicstopwords.arabicstopwords import stopwords_list
import nltk
from src.config.settings import MIN_PLATFORM_FREQUENCY, MIN_CONTENT_LENGTH, ANALYSIS_START_DATE

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
    """Minimal cleaning for transformers"""
    if pd.isna(text): 
        return ""

    # Basic cleaning
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace only
    
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

def _apply_text_cleaning(df, cleaning_func, suffix):
    """
    Helper function to apply text cleaning and reduce code duplication.
    
    Args:
        df: DataFrame with title and content columns
        cleaning_func: Function to apply for text cleaning
        suffix: Suffix for the created column names
    """
    df[f"title_clean_{suffix}"] = df["title"].astype(str).apply(cleaning_func)
    df[f"content_clean_{suffix}"] = df["content"].astype(str).apply(cleaning_func)
    
    # Create final text column with consistent naming
    if suffix == "agg":
        df["text_aggressive"] = df[f"title_clean_{suffix}"] + " " + df[f"content_clean_{suffix}"]
    elif suffix == "min":
        df["text_minimal"] = df[f"title_clean_{suffix}"] + " " + df[f"content_clean_{suffix}"]
    else:
        df[f"text_{suffix}"] = df[f"title_clean_{suffix}"] + " " + df[f"content_clean_{suffix}"]

def _validate_required_columns(df):
    """Validate that required columns exist in DataFrame"""
    required_columns = ['title', 'content']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

def prepare_data(file_path):
    """Complete data preparation with multiple cleaning approaches based on EDA findings"""
    
    # Load data if file path is provided
    if isinstance(file_path, str):
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples")
    else:
        # Backward compatibility - if DataFrame is passed directly
        df = file_path
    
    print("Standardizing columns and parsing dates...")
    # Rename columns for consistency (original CSV has these exact column names)
    if 'News content' in df.columns:
        df = df.rename(columns={"Id": "id", "News content": "content", "Label": "label"})
    
    # Validate required columns exist after renaming
    _validate_required_columns(df)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'label' in df.columns:
        df['label'] = df['label'].str.lower()

    print("Removing duplicates and filtering...")
    # Remove duplicates based on content and title (more specific than full row duplicates)
    original_len = len(df)
    df = df.drop_duplicates(subset=['content'])
    df = df.drop_duplicates(subset=['title'])
    print(f"Removed {original_len - len(df)} duplicates")
    
    # Filter by analysis period if date available
    if 'date' in df.columns:
        df = df[df['date'] >= ANALYSIS_START_DATE]
    
    # Group low-frequency platforms based on EDA findings
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        df['platform_grouped'] = df['platform'].apply(
            lambda x: x if platform_counts[x] >= MIN_PLATFORM_FREQUENCY else 'Other'
        )

    # Apply different text cleaning approaches for different model types
    print("Applying aggressive Arabic text cleaning...")
    _apply_text_cleaning(df, clean_arabic_text_aggressive, "agg")

    print("Applying minimal Arabic text cleaning...")
    _apply_text_cleaning(df, clean_arabic_text_minimal, "min")

    print("Applying transformers-ready text cleaning...")
    _apply_text_cleaning(df, clean_arabic_text_transformers, "transformers")

    print("Generating feature columns...")
    df["title_length"] = df["title"].apply(lambda x: len(str(x).split()))
    df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))

    # Remove outliers based on EDA analysis
    print("Removing outliers using IQR method...")
    original_len = len(df)
    df = remove_outliers_iqr(df, col="content_length")
    print(f"Removed {original_len - len(df)} outliers")

    # Encode labels and platforms
    if 'label' in df.columns:
        df['label'] = df['label'].map({'real': 0, 'fake': 1})
    if 'platform_grouped' in df.columns:
        df['platform_encoded'] = LabelEncoder().fit_transform(df['platform_grouped'])

    # Remove very short content based on EDA findings
    df = df[df['content_length'] > MIN_CONTENT_LENGTH].copy()
    
    print(f"Data cleaning completed - {len(df)} articles ready")
    return df
