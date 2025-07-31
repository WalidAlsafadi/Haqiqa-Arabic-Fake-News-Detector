"""
Updated hyperparameter tuning with proper train/validation/test splits.

This module implements hyperparameter tuning using validation set only,
ensuring that the test set remains untouched for final evaluation.
"""

import os
import json
import pickle
import pandas as pd
from scipy.sparse import vstack
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from src.config.settings import RANDOM_STATE

def get_param_grids():
    """Get parameter grids for hyperparameter tuning"""
    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1],
            'random_state': [RANDOM_STATE]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2],
            'random_state': [RANDOM_STATE]
        },
        'Logistic Regression': {
            'C': [1, 10],
            'max_iter': [500],
            'random_state': [RANDOM_STATE]
        },
        'SVM': {
            'C': [1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale'],
            'random_state': [RANDOM_STATE]
        }
    }
    return param_grids

def get_model_class(model_name):
    """Get model class for instantiation"""
    models = {
        'XGBoost': xgb.XGBClassifier,
        'Random Forest': RandomForestClassifier,
        'Logistic Regression': LogisticRegression,
        'SVM': SVC
    }
    return models.get(model_name)

def tune_best_models_proper(best_models, splits, output_dir="outputs/hyperparameter_tuning", force_retune=False):
    """
    Perform hyperparameter tuning on best models using validation set.
    
    ML Best Practices:
    1. Uses validation set for hyperparameter optimization
    2. Uses vectorizers fitted on training data only 
    3. Test set remains untouched for final evaluation
    4. Trains final models on train+validation combined
    
    Args:
        best_models: List of tuples [(model_name, dataset_name), ...]
        splits: Data splits from create_data_splits()
        output_dir: Directory to save tuning results
        force_retune: If True, ignore cached parameters and run fresh tuning
    
    Returns:
        dict: Tuned models and their performance
    """
    
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    print("Strategy: Validation-based hyperparameter optimization")
    print("Training data for fitting, validation for optimization, test held out")
    os.makedirs(output_dir, exist_ok=True)
    
    from src.utils.data_splits import get_train_data, get_val_data
    from src.models.model_selection import get_vectorizers_from_model_selection
    
    # Load vectorizers fitted during model selection
    vectorizers = get_vectorizers_from_model_selection()
    if not vectorizers:
        print("ERROR: No fitted vectorizers found from model selection!")
        print("Please run model selection first: python main.py --model-selection")
        return None
    
    param_grids = get_param_grids()
    tuned_results = {}
    
    for model_name, dataset_name in best_models:
        print(f"Tuning {model_name} on {dataset_name} dataset...")
        print("-" * 50)
        
        # Load cached results if available
        results_file = f"{output_dir}/{model_name}_{dataset_name}_best_params.json"
        model_file = f"{output_dir}/{model_name}_{dataset_name}_tuned_model.pkl"
        
        if os.path.exists(results_file) and os.path.exists(model_file) and not force_retune:
            print(f"  ✅ Loading cached results from {results_file}")
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            with open(model_file, 'rb') as f:
                saved_model = pickle.load(f)
            
            tuned_results[f"{model_name}_{dataset_name}"] = {
                **saved_results,
                'model': saved_model,
                'dataset': dataset_name
            }
            print(f"  📊 Loaded {model_name}_{dataset_name}: Validation F1={saved_results['best_score']:.4f}")
            continue
        elif force_retune and os.path.exists(results_file):
            print(f"  🔄 Force retune enabled, running fresh hyperparameter search...")
        
        # Get vectorizer for this model-dataset combination
        vectorizer_key = f"{model_name}_{dataset_name}"
        if vectorizer_key not in vectorizers:
            print(f"  ❌ No vectorizer found for {vectorizer_key}")
            continue
        
        vectorizer = vectorizers[vectorizer_key]
        print(f"  ✅ Using cached vectorizer: {vectorizer_key}")
        
        # Get data splits
        train_text, train_labels = get_train_data(splits, dataset_name)
        val_text, val_labels = get_val_data(splits, dataset_name)
        
        # Transform using the fitted vectorizer (no refitting!)
        X_train = vectorizer.transform(train_text.fillna(''))
        X_val = vectorizer.transform(val_text.fillna(''))
        
        print(f"  📊 Train set: {X_train.shape[0]} samples")
        print(f"  📊 Validation set: {X_val.shape[0]} samples")
        
        # Perform hyperparameter optimization
        try:
            model_class = get_model_class(model_name)
            if model_class is None:
                print(f"  ❌ Model {model_name} not supported for tuning")
                continue
            
            param_grid = param_grids.get(model_name, {})
            if not param_grid:
                print(f"  ❌ No parameter grid defined for {model_name}")
                continue
            
            print(f"  🔍 Testing {len(list(ParameterGrid(param_grid)))} parameter combinations...")
            
            best_score = 0
            best_params = None
            all_scores = []
            
            # Manual grid search using validation set
            for i, params in enumerate(ParameterGrid(param_grid)):
                # Train model with these parameters on training set
                model = model_class(**params)
                model.fit(X_train, train_labels)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_f1 = f1_score(val_labels, val_predictions, average='weighted')
                
                all_scores.append(val_f1)
                
                if val_f1 > best_score:
                    best_score = val_f1
                    best_params = params
                
                print(f"    Combination {i+1}: F1={val_f1:.4f}")
            
            print(f"  🎯 Best validation F1: {best_score:.4f}")
            print(f"  ⚙️  Best parameters: {best_params}")
            
            # Train final model with best parameters on train+validation combined
            print(f"  🔧 Training final model on train+validation combined...")
            final_model = model_class(**best_params)
            
            # Combine train and validation for final training
            X_train_val = vstack([X_train, X_val])
            y_train_val = pd.concat([train_labels, val_labels], ignore_index=True)
            
            final_model.fit(X_train_val, y_train_val)
            print(f"  ✅ Final model trained on {X_train_val.shape[0]} samples")
            
            # Save results
            results = {
                'model_name': model_name,
                'dataset': dataset_name,
                'best_params': best_params,
                'best_score': best_score,
                'tuning_method': 'validation_based',
                'train_size': X_train.shape[0],
                'val_size': X_val.shape[0],
                'final_train_size': X_train_val.shape[0]
            }
            
            # Save parameters to JSON (without model object)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save model separately
            with open(model_file, 'wb') as f:
                pickle.dump(final_model, f)
            
            tuned_results[f"{model_name}_{dataset_name}"] = {
                **results,
                'model': final_model
            }
            
            print(f"  💾 Results saved to {results_file}")
            print(f"  💾 Model saved to {model_file}")
            
        except Exception as e:
            print(f"  ❌ Hyperparameter tuning failed: {str(e)}")
            continue
        
        print()
    
    print("Hyperparameter tuning completed!")
    print("=" * 60)
    return tuned_results
