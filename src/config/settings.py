"""
Configuration settings for Palestine Fake News Detection Pipeline

This module contains all configuration parameters used throughout the ML pipeline.
All settings are derived from EDA analysis and proven to work with the current pipeline.
"""

# ==================== DATA PATHS ====================

DATASET_PATH = "data/raw/original_news_data.csv"

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
