"""
Data splitting utilities for consistent train/validation/test splits across the ML pipeline.

This ensures that:
1. The same test set is held out throughout the entire pipeline
2. TF-IDF is fitted only on training data
3. Hyperparameter tuning uses validation set
4. Final evaluation uses the held-out test set
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.settings import RANDOM_STATE

def create_data_splits(text_minimal, text_aggressive, y, save_splits=True):
    """
    Create consistent train/validation/test splits for both datasets.
    
    Split Strategy:
    - Train: 60% (for model training and cross-validation)
    - Validation: 20% (for hyperparameter tuning)
    - Test: 20% (for final evaluation only - held out)
    
    Args:
        text_minimal: Minimal cleaned text data
        text_aggressive: Aggressive cleaned text data  
        y: Target labels
        save_splits: Whether to save split indices for consistency
    
    Returns:
        dict: Contains all splits for both datasets
    """
    
    print("Creating consistent train/validation/test splits...")
    print("Split strategy: 60% train, 20% validation, 20% test")
    
    # First split: separate test set (20%)
    indices = range(len(y))
    train_val_indices, test_indices, y_train_val, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: train (60% of total) and validation (20% of total)
    train_indices, val_indices, y_train, y_val = train_test_split(
        train_val_indices, y_train_val, test_size=0.25,  # 0.25 * 0.8 = 0.2 of total
        random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    # Create splits for minimal dataset
    minimal_splits = {
        'train_text': text_minimal.iloc[train_indices].reset_index(drop=True),
        'val_text': text_minimal.iloc[val_indices].reset_index(drop=True), 
        'test_text': text_minimal.iloc[test_indices].reset_index(drop=True),
        'train_labels': y.iloc[train_indices].reset_index(drop=True),
        'val_labels': y.iloc[val_indices].reset_index(drop=True),
        'test_labels': y.iloc[test_indices].reset_index(drop=True)
    }
    
    # Create splits for aggressive dataset
    aggressive_splits = {
        'train_text': text_aggressive.iloc[train_indices].reset_index(drop=True),
        'val_text': text_aggressive.iloc[val_indices].reset_index(drop=True),
        'test_text': text_aggressive.iloc[test_indices].reset_index(drop=True), 
        'train_labels': y.iloc[train_indices].reset_index(drop=True),
        'val_labels': y.iloc[val_indices].reset_index(drop=True),
        'test_labels': y.iloc[test_indices].reset_index(drop=True)
    }
    
    splits = {
        'minimal': minimal_splits,
        'aggressive': aggressive_splits,
        'split_info': {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'total_size': len(y),
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }
    }
    
    print(f"Data splits created:")
    print(f"   Train: {len(train_indices)} samples ({len(train_indices)/len(y)*100:.1f}%)")
    print(f"   Validation: {len(val_indices)} samples ({len(val_indices)/len(y)*100:.1f}%)")
    print(f"   Test: {len(test_indices)} samples ({len(test_indices)/len(y)*100:.1f}%)")
    
    # Save splits for consistency across pipeline phases
    if save_splits:
        os.makedirs("data/processed/splits", exist_ok=True)
        with open("data/processed/splits/data_splits.pkl", 'wb') as f:
            pickle.dump(splits, f)
        print("Data splits saved to data/processed/splits/data_splits.pkl")
    
    return splits

def load_data_splits():
    """Load previously saved data splits"""
    splits_file = "data/processed/splits/data_splits.pkl"
    
    if os.path.exists(splits_file):
        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)
        
        info = splits['split_info']
        print("Loaded consistent data splits:")
        print(f"   Train: {info['train_size']} samples")
        print(f"   Validation: {info['val_size']} samples") 
        print(f"   Test: {info['test_size']} samples")
        
        return splits
    else:
        print("No saved data splits found")
        return None

def get_train_data(splits, dataset='minimal'):
    """Get training data for a specific dataset"""
    return splits[dataset]['train_text'], splits[dataset]['train_labels']

def get_val_data(splits, dataset='minimal'):
    """Get validation data for a specific dataset"""
    return splits[dataset]['val_text'], splits[dataset]['val_labels']

def get_test_data(splits, dataset='minimal'):
    """Get test data for a specific dataset (use only for final evaluation!)"""
    return splits[dataset]['test_text'], splits[dataset]['test_labels']

def get_train_val_data(splits, dataset='minimal'):
    """Get combined train+validation data (for model selection with cross-validation)"""
    train_text = splits[dataset]['train_text']
    val_text = splits[dataset]['val_text']
    train_labels = splits[dataset]['train_labels']
    val_labels = splits[dataset]['val_labels']
    
    # Combine train and validation
    combined_text = pd.concat([train_text, val_text], ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels], ignore_index=True)
    
    return combined_text, combined_labels
