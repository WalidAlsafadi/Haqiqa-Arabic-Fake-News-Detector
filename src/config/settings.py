"""
Configuration settings for Palestine Fake News Detection Pipeline

This module contains all configuration parameters used throughout the ML pipeline.
"""

# ==================== CORE SETTINGS ====================

# Logging
USE_PROFESSIONAL_LOGGING = True
LOG_LEVEL = "INFO"

# Data Processing
DATASETS_TO_USE = ["text_minimal", "text_aggressive", "text_transformers"]  # Column names after cleaning
REMOVE_OUTLIERS = True
MIN_TEXT_LENGTH = 10

# Model Training
CV_FOLDS = 5  # Cross-validation folds
RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF Features
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2    # Ignore rare words (appear in < 2 documents)
TFIDF_MAX_DF = 0.8  # Ignore common words (appear in > 80% of documents)

# ==================== HELPER FUNCTIONS ====================

def get_datasets():
    """Get list of datasets to process"""
    return [f"{dataset}.csv" for dataset in DATASETS_TO_USE]

def print_config():
    """Print current configuration"""
    print("PALESTINE FAKE NEWS DETECTION - CONFIGURATION")
    print("=" * 50)
    print(f"📊 Datasets: {DATASETS_TO_USE}")
    print(f"🔄 CV Folds: {CV_FOLDS}")
    print(f"🎯 Random State: {RANDOM_STATE}")
    print(f"📝 Max TF-IDF Features: {TFIDF_MAX_FEATURES}")
    print(f"📋 Min Document Frequency: {TFIDF_MIN_DF}")
    print(f"📋 Max Document Frequency: {TFIDF_MAX_DF}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
