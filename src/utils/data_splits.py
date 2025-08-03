import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.settings import RANDOM_STATE

class DataSplitter:
    """Simple and efficient data splitting with consistent indices"""
    
    def __init__(self, train_size=0.6, val_size=0.2, test_size=0.2, random_state=RANDOM_STATE):
        """
        Initialize DataSplitter with configurable split ratios.
        
        Args:
            train_size: Proportion for training (default: 0.6)
            val_size: Proportion for validation (default: 0.2) 
            test_size: Proportion for testing (default: 0.2)
            random_state: Random seed for reproducibility
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Split sizes must sum to 1.0")
            
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
    
    def create_splits(self, df, target='label', save_splits=True):
        """
        Create train/validation/test splits based on target distribution.
        
        Args:
            df: DataFrame with data to split
            target: Target column name for stratification
            save_splits: Whether to save split indices to disk
        """
        print(f"Creating stratified splits: {self.train_size:.0%} train, {self.val_size:.0%} val, {self.test_size:.0%} test")
        
        y = df[target]
        indices = list(range(len(df)))
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_ratio = self.val_size / (self.train_size + self.val_size)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio, random_state=self.random_state,
            stratify=y.iloc[train_val_indices]
        )
        
        self.train_indices = sorted(train_indices)
        self.val_indices = sorted(val_indices)
        self.test_indices = sorted(test_indices)
        
        print(f"Split created: {len(self.train_indices)} train, {len(self.val_indices)} val, {len(self.test_indices)} test")
        
        if save_splits:
            self._save_splits()
    
    def get_train(self, df, text_column='text_minimal', target='label'):
        """Get training data for specified text column"""
        self._check_splits_exist()
        return (df[text_column].iloc[self.train_indices].reset_index(drop=True),
                df[target].iloc[self.train_indices].reset_index(drop=True))
    
    def get_val(self, df, text_column='text_minimal', target='label'):
        """Get validation data for specified text column"""
        self._check_splits_exist()
        return (df[text_column].iloc[self.val_indices].reset_index(drop=True),
                df[target].iloc[self.val_indices].reset_index(drop=True))
    
    def get_test(self, df, text_column='text_minimal', target='label'):
        """Get test data for specified text column (use only for final evaluation!)"""
        self._check_splits_exist()
        return (df[text_column].iloc[self.test_indices].reset_index(drop=True),
                df[target].iloc[self.test_indices].reset_index(drop=True))
    
    def get_train_val(self, df, text_column='text_minimal', target='label'):
        """Get combined train+validation data (useful for cross-validation)"""
        self._check_splits_exist()
        combined_indices = sorted(self.train_indices + self.val_indices)
        return (df[text_column].iloc[combined_indices].reset_index(drop=True),
                df[target].iloc[combined_indices].reset_index(drop=True))
    
    
    def get_train_from_csv(self, csv_path, text_column='text', target='label'):
        """Get training data from CSV file using saved indices"""
        self._check_splits_exist()
        df = pd.read_csv(csv_path)
        return (df[text_column].iloc[self.train_indices].reset_index(drop=True),
                df[target].iloc[self.train_indices].reset_index(drop=True))
    
    def get_val_from_csv(self, csv_path, text_column='text', target='label'):
        """Get validation data from CSV file using saved indices"""
        self._check_splits_exist()
        df = pd.read_csv(csv_path)
        return (df[text_column].iloc[self.val_indices].reset_index(drop=True),
                df[target].iloc[self.val_indices].reset_index(drop=True))
    
    def get_test_from_csv(self, csv_path, text_column='text', target='label'):
        """Get test data from CSV file using saved indices (use only for final evaluation!)"""
        self._check_splits_exist()
        df = pd.read_csv(csv_path)
        return (df[text_column].iloc[self.test_indices].reset_index(drop=True),
                df[target].iloc[self.test_indices].reset_index(drop=True))
    
    def get_train_val_from_csv(self, csv_path, text_column='text', target='label'):
        """Get combined train+validation data from CSV file using saved indices"""
        self._check_splits_exist()
        df = pd.read_csv(csv_path)
        combined_indices = sorted(self.train_indices + self.val_indices)
        return (df[text_column].iloc[combined_indices].reset_index(drop=True),
                df[target].iloc[combined_indices].reset_index(drop=True))
    
    def _check_splits_exist(self):
        """Check if splits have been created"""
        if self.train_indices is None:
            raise ValueError("Splits not created. Call create_splits() first.")
    
    def _save_splits(self):
        """Save split indices to disk for consistency"""
        os.makedirs("data/processed/splits", exist_ok=True)
        
        split_data = {
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices,
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'random_state': self.random_state
        }
        
        with open("data/processed/splits/split_indices.pkl", 'wb') as f:
            pickle.dump(split_data, f)
        print("Split indices saved to data/processed/splits/split_indices.pkl")
    
    @classmethod
    def load_splits(cls):
        """Load previously saved split indices"""
        splits_file = "data/processed/splits/split_indices.pkl"
        
        if not os.path.exists(splits_file):
            print("No saved split indices found")
            return None
        
        with open(splits_file, 'rb') as f:
            split_data = pickle.load(f)
        
        # Create instance with saved parameters
        splitter = cls(
            train_size=split_data['train_size'],
            val_size=split_data['val_size'], 
            test_size=split_data['test_size'],
            random_state=split_data['random_state']
        )
        
        # Load the indices
        splitter.train_indices = split_data['train_indices']
        splitter.val_indices = split_data['val_indices']
        splitter.test_indices = split_data['test_indices']
        
        print("Loaded split indices:")
        print(f"   Train: {len(splitter.train_indices)} samples")
        print(f"   Validation: {len(splitter.val_indices)} samples")
        print(f"   Test: {len(splitter.test_indices)} samples")
        
        return splitter
