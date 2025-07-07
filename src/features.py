import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# TF-IDF Vectorizer Builder
# ------------------------------

def build_tfidf_vectorizer(texts, max_features=5000, ngram_range=(1,2)):
    """
    Fit a TF-IDF vectorizer on training text data.
    
    Args:
        texts (pd.Series or list): Text data
        max_features (int): Max vocab size
        ngram_range (tuple): N-gram range (default unigrams + bigrams)
    
    Returns:
        vectorizer: Fitted TfidfVectorizer
        X: Sparse matrix (text transformed)
    """
    print("🔹 Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer='word'
    )
    X = vectorizer.fit_transform(texts)
    print(f"✅ TF-IDF shape: {X.shape}")
    return vectorizer, X

# ------------------------------
# Apply Existing Vectorizer
# ------------------------------

def transform_texts(vectorizer, texts):
    """
    Transform new texts using a previously fitted vectorizer.
    
    Args:
        vectorizer: fitted TfidfVectorizer
        texts: list or pd.Series of text
    
    Returns:
        Sparse matrix
    """
    return vectorizer.transform(texts)

# ------------------------------
# Save / Load Vectorizer
# ------------------------------

def save_vectorizer(vectorizer, path="models/tfidf_vectorizer.pkl"):
    """
    Save a fitted vectorizer to disk using joblib.
    """
    joblib.dump(vectorizer, path)
    print(f"📦 Saved vectorizer to: {path}")

def load_vectorizer(path="models/tfidf_vectorizer.pkl"):
    """
    Load a saved vectorizer.
    """
    print(f"🔄 Loading vectorizer from: {path}")
    return joblib.load(path)

# ------------------------------
# Standalone Test (Optional)
# ------------------------------

if __name__ == "__main__":
    # Test run for development/debugging
    df = pd.read_csv("data/processed/text_clean_agg.csv")
    vectorizer, X = build_tfidf_vectorizer(df["text_clean_agg"])
    save_vectorizer(vectorizer)
    print("✅ Vectorizer built and saved.")