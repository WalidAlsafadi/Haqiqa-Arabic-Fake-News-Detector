"""
Modern Data Preparation Module for Palestine Fake News Detection Project

Professional NLP pipeline following industry best practices for Arabic text processing.

Features:
- Multiple text cleaning strategies (minimal, aggressive, transformers)
- Efficient merging of multiple datasets  
- Modern Arabic text preprocessing techniques
- Consistent data splitting and validation
"""

import os
import re
import pandas as pd
import nltk
from sklearn.utils import shuffle
from arabicstopwords.arabicstopwords import stopwords_list
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Optimized stopwords set for performance
ARABIC_STOPWORDS = set(stopwords_list())


def get_project_path(relative_path):
    """Get absolute path relative to project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    if relative_path.startswith('../'):
        relative_path = relative_path[3:]
    
    return os.path.join(project_root, relative_path)


def normalize_arabic(text):
    """Normalize Arabic text by standardizing characters."""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    return text


def remove_diacritics(text):
    """Remove Arabic diacritics."""
    return re.sub(r'[\u064B-\u0652]', '', text)


def clean_arabic_text_minimal(text):
    """
    Minimal Arabic text cleaning - keeps more context for ML models.
    Less aggressive approach that preserves more linguistic features.
    """
    if pd.isna(text): 
        return ""
    
    text = str(text).strip().lower()
    text = normalize_arabic(text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep only Arabic and basic chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()


def clean_arabic_text_aggressive(text):
    """
    Aggressive Arabic text cleaning with stopword removal for ML models.
    More thorough preprocessing that removes noise and common words.
    """
    if pd.isna(text): 
        return ""
    
    text = str(text).strip().lower()
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep only Arabic and basic chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Remove stopwords and short tokens
    if text:
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in ARABIC_STOPWORDS and len(token) > 1]
        text = ' '.join(tokens)
    
    return text.strip()


def clean_arabic_text_transformers(text):
    """
    Modern minimal cleaning for transformer models (follows HuggingFace best practices).
    Transformers work better with less aggressive preprocessing.
    """
    if pd.isna(text): 
        return ""

    text = str(text).strip()
    
    # Only essential cleaning for transformers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)       # Remove mentions/hashtags
    text = re.sub(r'\s+', ' ', text)            # Normalize whitespace
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep Arabic and basic chars
    
    return text.strip()


def standardize_columns(df, dataset_type='ucas'):
    """
    Standardize column names and structure for consistency.
    """
    if dataset_type.lower() == 'ucas':
        # UCAS dataset: merge title + content into single content column
        column_mapping = {
            "Id": "id",
            "News content": "content", 
            "Label": "label"
        }
        df = df.rename(columns=column_mapping)
        
        # Merge title + content for UCAS (following NLP best practices)
        if 'title' not in df.columns and 'id' in df.columns:
            df['title'] = df['id'].astype(str)  # Use ID as title if needed
        
        if 'title' in df.columns:
            df['content'] = df['title'].astype(str) + " " + df['content'].astype(str)
            
    elif dataset_type.lower() == 'kaggle':
        # Kaggle dataset: use content only (no title column needed)
        column_mapping = {
            "Article_content": "content",
            "Label": "label"
        }
        df = df.rename(columns=column_mapping)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Keep only essential columns
    df = df[['content', 'label']].copy()
    
    return df


def load_and_standardize_dataset(file_path, dataset_type):
    """
    Load and standardize a single dataset with proper column mapping.
    """
    print(f"Loading {dataset_type.upper()} dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  → Loaded {len(df)} samples")
    
    # Standardize columns based on dataset type
    df = standardize_columns(df, dataset_type=dataset_type)
    
    # Remove internal duplicates
    print(f"  → Removing duplicates...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['content'], keep='first')
    removed = original_len - len(df)
    print(f"  → Removed {removed} duplicates, {len(df)} unique samples")
    
    # Add dataset source for tracking
    df['source'] = dataset_type.upper()
    
    return df


def merge_datasets(ucas_path=None, kaggle_path=None):
    """
    Intelligently merge UCAS and Kaggle datasets with proper handling.
    """
    datasets = []
    
    # Load UCAS dataset
    if ucas_path:
        ucas_df = load_and_standardize_dataset(ucas_path, 'ucas')
        datasets.append(ucas_df)
    
    # Load Kaggle dataset
    if kaggle_path:
        kaggle_df = load_and_standardize_dataset(kaggle_path, 'kaggle')
        datasets.append(kaggle_df)
    
    if not datasets:
        raise ValueError("At least one dataset path must be provided")
    
    # Merge datasets
    if len(datasets) == 1:
        combined_df = datasets[0]
        print(f"Using single dataset: {len(combined_df)} samples")
    else:
        print("Merging datasets...")
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"  → Combined: {len(combined_df)} samples")
        
        # Show distribution by source
        source_counts = combined_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  → {source}: {count} samples")
    
    return combined_df


def prepare_modern_dataset(ucas_path=None, kaggle_path=None):
    """
    Modern data preparation pipeline following NLP best practices.
    
    Args:
        ucas_path: Path to UCAS NLP Course 2025 dataset (Palestinian news, academic collaboration)
        kaggle_path: Path to Arabic News Verification dataset (scraped from fact-checking websites)
        
    Returns:
        pandas.DataFrame: Prepared and standardized dataset ready for ML pipeline
    """
    # Convert relative paths to absolute
    if ucas_path and not os.path.isabs(ucas_path):
        ucas_path = get_project_path(ucas_path)
    if kaggle_path and not os.path.isabs(kaggle_path):
        kaggle_path = get_project_path(kaggle_path)
    
    # Load and merge datasets
    df = merge_datasets(ucas_path, kaggle_path)
    # Shuffle rows to mix UCAS and Kaggle samples using config random state
    from src.config.settings import RANDOM_STATE
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("Shuffled merged dataset to mix samples from all sources")
    
    # Validate required columns
    required_columns = ['content', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    # Check for cross-dataset duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['content'], keep='first')
    removed = original_len - len(df)
    if removed > 0:
        print(f"Removed {removed} cross-dataset duplicates")

    # Apply different text cleaning approaches for different model types
    print("Applying text cleaning...")
    df['text_minimal'] = df['content'].apply(clean_arabic_text_minimal)
    df['text_aggressive'] = df['content'].apply(clean_arabic_text_aggressive)
    df['text_transformers'] = df['content'].apply(clean_arabic_text_transformers)

    # Compute text statistics
    df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))
    df["text_length"] = df["text_minimal"].apply(lambda x: len(str(x).split()))

    # Remove outliers using IQR method
    original_len = len(df)
    Q1 = df['content_length'].quantile(0.25)
    Q3 = df['content_length'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df['content_length'] >= lower) & (df['content_length'] <= upper)]
    removed = original_len - len(df)
    if removed > 0:
        print(f"Removed {removed} outliers")

    # Map label column to integers for ML compatibility
    if 'label' in df.columns:
        df['label'] = df['label'].map({'real': 0, 'fake': 1})
        # If any unmapped values remain, raise error for transparency
        if df['label'].isnull().any():
            raise ValueError("Label column contains values other than 'real' or 'fake'.")

    print(f"Data preparation completed: {len(df)} articles ready")
    return df


def create_separate_datasets(df):
    """
    Create separate datasets for different cleaning approaches.
    """
    # Create minimal dataset (transformer-optimized)
    minimal_df = df[['text_minimal', 'label']].copy()
    minimal_df = minimal_df.rename(columns={'text_minimal': 'text'})
    minimal_df['text'] = minimal_df['text'].fillna("").astype(str)

    aggressive_df = df[['text_aggressive', 'label']].copy()
    aggressive_df = aggressive_df.rename(columns={'text_aggressive': 'text'})
    aggressive_df['text'] = aggressive_df['text'].fillna("").astype(str)

    transformers_df = df[['text_transformers', 'label']].copy()
    transformers_df = transformers_df.rename(columns={'text_transformers': 'text'})
    transformers_df['text'] = transformers_df['text'].fillna("").astype(str)

    return minimal_df, aggressive_df, transformers_df


def save_dataset(df, output_path):
    """Save dataset to specified path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')


def prepare_datasets(include_arabic_verification=True, save_to_processed=True):
    """
    Simple function to prepare datasets with modern NLP techniques.
    
    Args:
        include_arabic_verification (bool): Whether to include Arabic News Verification dataset
        save_to_processed (bool): Whether to save to data/processed/
        
    Returns:
        tuple: (minimal_df, aggressive_df, transformers_df)
    """
    # Import config values
    from src.config.settings import UCAS_NLP_COURSE_DATASET_PATH, ARABIC_NEWS_VERIFICATION_DATASET_PATH
    
    # Define dataset paths using config
    ucas_path = UCAS_NLP_COURSE_DATASET_PATH
    kaggle_path = ARABIC_NEWS_VERIFICATION_DATASET_PATH if include_arabic_verification else None
    
    # Check if Arabic News Verification dataset exists
    if kaggle_path:
        kaggle_full_path = get_project_path(kaggle_path)
        if not os.path.exists(kaggle_full_path):
            print(f"Arabic News Verification dataset not found, proceeding with UCAS only...")
            kaggle_path = None
    
    # Prepare dataset
    df = prepare_modern_dataset(ucas_path, kaggle_path)
    
    # Create separate datasets
    minimal_df, aggressive_df, transformers_df = create_separate_datasets(df)
    
    # Save if requested
    if save_to_processed:
        from src.config.settings import MINIMAL_DATASET_PATH, AGGRESSIVE_DATASET_PATH, TRANSFORMERS_DATASET_PATH
        save_dataset(minimal_df, get_project_path(MINIMAL_DATASET_PATH))
        save_dataset(aggressive_df, get_project_path(AGGRESSIVE_DATASET_PATH))
        save_dataset(transformers_df, get_project_path(TRANSFORMERS_DATASET_PATH))
        print(f"Datasets saved to processed directory ({len(df)} samples each)")
    
    return minimal_df, aggressive_df, transformers_df


def main():
    """
    Main execution function for data preparation pipeline.
    """
    print("Palestine Fake News Detection - Data Preparation")
    print("=" * 50)
    
    try:
        # Import config values
        from src.config.settings import UCAS_NLP_COURSE_DATASET_PATH, ARABIC_NEWS_VERIFICATION_DATASET_PATH
        
        # Define dataset paths using config
        ucas_path = UCAS_NLP_COURSE_DATASET_PATH
        kaggle_path = ARABIC_NEWS_VERIFICATION_DATASET_PATH
        
        # Check if Arabic News Verification dataset exists
        kaggle_full_path = get_project_path(kaggle_path)
        if not os.path.exists(kaggle_full_path):
            print(f"Arabic News Verification dataset not found, proceeding with UCAS only...")
            kaggle_path = None
        
        # Prepare dataset
        df = prepare_modern_dataset(ucas_path, kaggle_path)
        
        # Create separate datasets for different model types
        minimal_df, aggressive_df, transformers_df = create_separate_datasets(df)

        # Save datasets using config paths
        from src.config.settings import MINIMAL_DATASET_PATH, AGGRESSIVE_DATASET_PATH, TRANSFORMERS_DATASET_PATH
        print("Saving datasets...")
        save_dataset(minimal_df, get_project_path(MINIMAL_DATASET_PATH))
        save_dataset(aggressive_df, get_project_path(AGGRESSIVE_DATASET_PATH))
        save_dataset(transformers_df, get_project_path(TRANSFORMERS_DATASET_PATH))
        
        print(f"  → Minimal: {len(minimal_df)} samples")
        print(f"  → Aggressive: {len(aggressive_df)} samples")
        print(f"  → Transformers: {len(transformers_df)} samples")

        print("=" * 50)
        print("✅ Data preparation completed successfully")
        print(f"✅ Final dataset: {len(df)} samples ready for training")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
