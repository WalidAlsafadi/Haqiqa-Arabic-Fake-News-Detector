"""
Configuration settings for Palestine Fake News Detection Pipeline

This module contains all configuration parameters used throughout the ML pipeline.
All settings are derived from EDA analysis and proven to work with the current pipeline.
"""

import os
from pathlib import Path

# ==================== DATASET DOCUMENTATION ====================

"""
DATASET INFORMATION:

1. UCAS NLP Course 2025 Dataset (ucas_nlp_course_2025/):
   - Source: Student collaboration with Dr. Tareq Altalmas, UCAS NLP Course, Spring 2025
   - Collection: Each student group responsible for specific months of data collection
   - Status: Not officially published or shared yet (academic research in progress)
   - Content: Palestinian news articles with fake/real labels
   - Academic Context: University research project with supervised data collection

2. Arabic News Verification Scraped Dataset (arabic_news_verification_scraped/):
   - Source: Kaggle dataset by David Ozil
   - URL: https://www.kaggle.com/datasets/shyakanobledavid/david-ozil
   - Collection Method: Scraped from popular Arabic news verification websites:
     * Misbar (https://misbar.com/)
     * No Rumors (http://norumors.net/) 
     * Verify-Sy (https://verify-sy.com/)
     * Fatabyyano (https://fatabyyano.net/)
   - Content: Arabic news articles with preprocessing (cleaning, tokenization)
   - Columns: Article_content (Arabic text), Topic (news category), Label (fake/real)
   - Status: Publicly available on Kaggle platform

3. Additional Research Datasets:
   - araieval_2024/: ArAIEval 2024 competition dataset
   - wanlp2022_propaganda/: WANLP 2022 propaganda detection dataset
"""

# ==================== PROJECT PATHS ====================

# Base directories
DATA_DIR = "data"
SRC_DIR = "src"
OUTPUTS_DIR = "outputs"
SAVED_MODELS_DIR = "saved_models"
NOTEBOOKS_DIR = "notebooks"

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(PROCESSED_DATA_DIR, "splits")

# Dataset paths with professional naming
UCAS_NLP_COURSE_DATASET_PATH = os.path.join(RAW_DATA_DIR, "ucas_nlp_course_2025", "ucas_nlp_course_dataset.csv")
ARABIC_NEWS_VERIFICATION_DATASET_PATH = os.path.join(RAW_DATA_DIR, "arabic_news_verification_scraped", "arabic_news_verification.csv")

# Processed dataset paths
MINIMAL_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "minimal_cleaned.csv")
AGGRESSIVE_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "aggressive_cleaned.csv")
TRANSFORMERS_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "transformers_cleaned.csv")

# Dataset mapping for easy access with professional naming
DATASET_PATHS = {
    'minimal': MINIMAL_DATASET_PATH,
    'aggressive': AGGRESSIVE_DATASET_PATH, 
    'transformers': TRANSFORMERS_DATASET_PATH
}

# Raw dataset mapping with descriptive names
RAW_DATASET_PATHS = {
    'ucas_nlp_course_2025': UCAS_NLP_COURSE_DATASET_PATH,
    'arabic_news_verification_scraped': ARABIC_NEWS_VERIFICATION_DATASET_PATH
}

# Dataset metadata for documentation and validation
DATASET_METADATA = {
    'ucas_nlp_course_2025': {
        'description': 'UCAS NLP Course 2025 - Palestinian News Dataset',
        'source': 'Dr. Tareq Altalmas NLP Course Collaboration',
        'status': 'Academic Research (Not Published)',
        'collection_period': 'Spring 2025',
        'collection_method': 'Student groups by monthly assignments'
    },
    'arabic_news_verification_scraped': {
        'description': 'Arabic News Verification Dataset from Fact-Checking Websites',
        'source': 'Kaggle - David Ozil',
        'status': 'Publicly Available',
        'collection_method': 'Web scraping from verification platforms',
        'platforms': ['Misbar', 'No Rumors', 'Verify-Sy', 'Fatabyyano']
    }
}

# ==================== MODEL PATHS ====================

# Model directories
TRADITIONAL_MODELS_DIR = os.path.join(SAVED_MODELS_DIR, "traditional")
ARABERT_MODELS_DIR = os.path.join(SAVED_MODELS_DIR, "arabert")

# Model files
BEST_TRADITIONAL_MODEL_PATH = os.path.join(TRADITIONAL_MODELS_DIR, "best_model.pkl")
ARABERT_MODEL_PATH = ARABERT_MODELS_DIR

# ==================== OUTPUT PATHS ====================

# Output directories
MODEL_SELECTION_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "model_selection")
HYPERPARAMETER_TUNING_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "hyperparameter_tuning")
FINAL_EVALUATION_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "final_evaluation")
ARABERT_EVALUATION_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "arabert_evaluation")

# ==================== MODEL TRAINING ====================

# Cross-validation and reproducibility
CV_FOLDS = 5
RANDOM_STATE = 42

# TF-IDF Vectorization (optimized for Arabic text)
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2    # Ignore rare words (appear in < 2 documents)
TFIDF_MAX_DF = 0.8  # Ignore common words (appear in > 80% of documents)

# ==================== TEXT PREPROCESSING ====================

# Preprocessing Constants (derived from EDA analysis)
MIN_PLATFORM_FREQUENCY = 30      # Minimum frequency to keep platform separate
MIN_CONTENT_LENGTH = 10          # Minimum content length to keep article
ANALYSIS_START_DATE = '2023-01-01'  # Start date for analysis period

# ==================== ARABERT CONFIGURATION ====================

# AraBERT model configuration
ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
ARABERT_MAX_LENGTH = 512
ARABERT_NUM_LABELS = 2

# AraBERT training parameters
ARABERT_EPOCHS = 7
ARABERT_BATCH_SIZE = 16
ARABERT_LEARNING_RATE = 2e-5
ARABERT_WEIGHT_DECAY = 0.01
ARABERT_WARMUP_STEPS = 500
ARABERT_LOGGING_STEPS = 100
ARABERT_SAVE_TOTAL_LIMIT = 1

def get_dataset_info(dataset_key: str = None):
    """
    Get professional dataset information for documentation and validation.
    
    Args:
        dataset_key: Specific dataset key ('ucas_nlp_course_2025' or 'arabic_news_verification_scraped')
                    If None, returns information for all datasets
    
    Returns:
        Dictionary with dataset metadata or single dataset info
    """
    if dataset_key:
        return DATASET_METADATA.get(dataset_key, {})
    return DATASET_METADATA

def print_dataset_info():
    """Print formatted dataset information for project documentation."""
    print("\n" + "="*70)
    print("PALESTINE FAKE NEWS DETECTION - DATASET INFORMATION")
    print("="*70)
    
    for key, metadata in DATASET_METADATA.items():
        print(f"\n📊 {metadata['description']}")
        print(f"   Source: {metadata['source']}")
        print(f"   Status: {metadata['status']}")
        if 'collection_period' in metadata:
            print(f"   Collection Period: {metadata['collection_period']}")
        if 'collection_method' in metadata:
            print(f"   Collection Method: {metadata['collection_method']}")
        if 'platforms' in metadata:
            print(f"   Platforms: {', '.join(metadata['platforms'])}")
    
    print("\n" + "="*70)

# ==================== UTILITY FUNCTIONS ====================

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent

def ensure_dir_exists(path: str):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)
    return path

def get_absolute_path(relative_path: str):
    """Convert relative path to absolute path from project root"""
    return os.path.join(get_project_root(), relative_path)

# ==================== VALIDATION ====================

def validate_paths():
    """Validate that all required directories exist or can be created"""
    required_dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
        SAVED_MODELS_DIR, TRADITIONAL_MODELS_DIR, ARABERT_MODELS_DIR,
        OUTPUTS_DIR, MODEL_SELECTION_OUTPUT_DIR, HYPERPARAMETER_TUNING_OUTPUT_DIR,
        FINAL_EVALUATION_OUTPUT_DIR, ARABERT_EVALUATION_OUTPUT_DIR
    ]
    
    for dir_path in required_dirs:
        ensure_dir_exists(dir_path)
    
    return True
