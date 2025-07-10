import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from arabicstopwords.arabicstopwords import stopwords_list

# Arabic stopwords and stemmer setup
ARABIC_STOPWORDS = set(stopwords_list())
stemmer = ISRIStemmer()

# ------------------- #
# TEXT CLEANING UTILS #
# ------------------- #

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    return text

def remove_diacritics(text: str) -> str:
    return re.sub(r'[\u064B-\u0652]', '', text)

def remove_non_arabic(text: str) -> str:
    return re.sub(r"[^\u0600-\u06FF\s]", " ", text)

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

# --------------------- #
# FULL CLEAN PIPELINE   #
# --------------------- #

def clean_arabic_text_minimal(text: str) -> str:
    """Light Arabic cleaning — normalize only."""
    if pd.isna(text): return ""
    text = text.lower()
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_arabic_text(text: str) -> str:
    """Full Arabic NLP cleaning: normalize, tokenize, remove stopwords, stem."""
    if pd.isna(text): return ""
    text = text.lower()
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
    tokens = stem_tokens(tokens)

    return " ".join(tokens)
