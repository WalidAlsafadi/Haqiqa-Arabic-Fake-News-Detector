import os
import json
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

from src.config.settings import RANDOM_STATE
from src.utils.data_splits import DataSplitter
from src.models.model_selection import load_best_pipeline, get_best_model

def get_param_grids():
    """Get parameter grids for hyperparameter tuning with proper class imbalance handling"""
    param_grids = {
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.1, 0.2],
            'classifier__scale_pos_weight': [2.47],  # Handle class imbalance
            'classifier__random_state': [RANDOM_STATE]
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': ['balanced'],  # Handle class imbalance
            'classifier__random_state': [RANDOM_STATE]
        },
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10],
            'classifier__class_weight': ['balanced'],  # Handle class imbalance
            'classifier__max_iter': [500],
            'classifier__random_state': [RANDOM_STATE]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__class_weight': ['balanced'],  # Handle class imbalance
            'classifier__gamma': ['scale'],
            'classifier__random_state': [RANDOM_STATE]
        },
        'Naive Bayes': {
            # Naive Bayes has no significant hyperparameters to tune
            # Using empty dict - will skip tuning but still work
        }
    }
    return param_grids

def _validate_inputs(splitter):
    """Validate inputs for hyperparameter tuning"""
    if splitter.train_indices is None:
        raise ValueError("DataSplitter has no splits. Call create_splits() first.")

def _load_cached_results(output_dir, model_name, dataset_name):
    """Load cached tuning results if available"""
    results_file = f"{output_dir}/{model_name}_{dataset_name}_best_params.json"
    model_file = f"{output_dir}/{model_name}_{dataset_name}_tuned_model.pkl"
    
    if os.path.exists(results_file) and os.path.exists(model_file):
        try:
            with open(results_file, 'r') as f:
                cached_results = json.load(f)
            with open(model_file, 'rb') as f:
                cached_model = pickle.load(f)
            
            print(f"Loading cached results: F1={cached_results['best_score']:.4f}")
            return cached_results, cached_model
        except Exception as e:
            print(f"Failed to load cached results: {str(e)}")
            return None
    return None

def _perform_grid_search(baseline_pipeline, param_grid, train_text, train_labels, val_text, val_labels, model_name):
    """Perform grid search using validation set"""
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    best_score = 0
    best_params = None
    best_pipeline = None
    
    for i, params in enumerate(param_combinations):
        try:
            # Create new pipeline with these parameters
            # Start with a copy of the baseline pipeline to preserve TF-IDF settings
            current_pipeline = Pipeline([
                ('tfidf', baseline_pipeline.named_steps['tfidf']),
                ('classifier', baseline_pipeline.named_steps['classifier'].__class__(**baseline_pipeline.named_steps['classifier'].get_params()))
            ])
            current_pipeline.set_params(**params)
            
            # Train on training set
            current_pipeline.fit(train_text.fillna(''), train_labels)
            
            # Evaluate on validation set
            val_predictions = current_pipeline.predict(val_text.fillna(''))
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            
            if val_f1 > best_score:
                best_score = val_f1
                best_params = params
                best_pipeline = current_pipeline
            
        except Exception as e:
            continue
    
    return best_score, best_params, best_pipeline

def _train_final_model(baseline_pipeline, best_params, train_val_text, train_val_labels):
    """Train final model on combined train+validation data"""
    final_pipeline = Pipeline([
        ('tfidf', baseline_pipeline.named_steps['tfidf']),
        ('classifier', baseline_pipeline.named_steps['classifier'].__class__(**baseline_pipeline.named_steps['classifier'].get_params()))
    ])
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(train_val_text.fillna(''), train_val_labels)
    
    return final_pipeline

def _save_tuning_results(results, final_pipeline, output_dir, model_name, dataset_name):
    """Save tuning results and final model"""
    try:
        results_file = f"{output_dir}/{model_name}_{dataset_name}_best_params.json"
        model_file = f"{output_dir}/{model_name}_{dataset_name}_tuned_model.pkl"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(model_file, 'wb') as f:
            pickle.dump(final_pipeline, f)
        
        return True
    except Exception as e:
        print(f"Failed to save results: {str(e)}")
        return False

def tune_best_model(splitter, output_dir="outputs/hyperparameter_tuning"):
    """
    Perform hyperparameter tuning on the best model from model selection.
    
    ML Best Practices:
    1. Uses only the best model from model selection
    2. Uses validation set for hyperparameter optimization
    3. Test set remains untouched for final evaluation
    4. Pipeline approach with TF-IDF + model tuning
    
    Args:
        splitter: DataSplitter instance with loaded splits
        output_dir: Directory to save tuning results
    
    Returns:
        tuple: (results dict, final_pipeline) or None if failed
    """
    
    # Validate inputs
    _validate_inputs(splitter)
    
    print("\nHYPERPARAMETER TUNING")
    print("=" * 40)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load best model info from model selection
    best_model_info = get_best_model()
    if not best_model_info:
        print("No model selection results found!")
        print("Please run model selection first")
        return None
    
    model_name = best_model_info['model_name']
    dataset_name = best_model_info['dataset']
    
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Baseline CV F1: {best_model_info['cv_f1_mean']:.4f}")
    print(f"Baseline Val F1: {best_model_info['val_f1_score']:.4f}")
    
    # Check for cached results
    cached_result = _load_cached_results(output_dir, model_name, dataset_name)
    if cached_result:
        return cached_result
    
    # Load baseline pipeline
    baseline_pipeline, _ = load_best_pipeline()
    if baseline_pipeline is None:
        print("Could not load best pipeline!")
        return None
    
    # Dataset file mapping
    dataset_files = {
        'minimal': 'data/processed/minimal_cleaned.csv',
        'aggressive': 'data/processed/aggressive_cleaned.csv'
    }
    
    # Get training and validation data from CSV
    csv_path = dataset_files[dataset_name]
    if not os.path.exists(csv_path):
        print(f"Dataset file not found: {csv_path}")
        return None
        
    train_text, train_labels = splitter.get_train_from_csv(csv_path)
    val_text, val_labels = splitter.get_val_from_csv(csv_path)
    
    # Get parameter grid for this model
    param_grids = get_param_grids()
    param_grid = param_grids.get(model_name, {})
    
    if not param_grid:
        print(f"No parameter grid defined for {model_name}")
        return None
    
    # Handle models with no tunable parameters (like Naive Bayes)
    if len(param_grid) == 0:
        print(f"{model_name} has no hyperparameters to tune")
        
        # Train baseline on train+validation data
        train_val_text, train_val_labels = splitter.get_train_val_from_csv(csv_path)
        baseline_pipeline.fit(train_val_text.fillna(''), train_val_labels)
        
        # Save results showing no improvement (no tuning performed)
        results = {
            'model_name': model_name,
            'dataset': dataset_name,
            'best_params': {},
            'best_score': best_model_info['val_f1_score'],  # Same as baseline
            'baseline_score': best_model_info['val_f1_score'],
            'improvement': 0.0,  # No improvement possible
            'train_size': len(train_text),
            'val_size': len(val_text),
            'final_train_size': len(train_val_text)
        }
        
        if _save_tuning_results(results, baseline_pipeline, output_dir, model_name, dataset_name):
            return results, baseline_pipeline
        else:
            print("Failed to save results")
            return None
    
    # Perform grid search
    best_score, best_params, best_pipeline = _perform_grid_search(
        baseline_pipeline, param_grid, train_text, train_labels, val_text, val_labels, model_name
    )
    
    if best_pipeline is None:
        print("All parameter combinations failed!")
        return None
    
    print(f"Best validation F1: {best_score:.4f}")
    print(f"Improvement: {best_score - best_model_info['val_f1_score']:.4f}")
    
    # Train final model on train+validation combined
    train_val_text, train_val_labels = splitter.get_train_val_from_csv(csv_path)
    final_pipeline = _train_final_model(baseline_pipeline, best_params, train_val_text, train_val_labels)
    
    # Save results
    results = {
        'model_name': model_name,
        'dataset': dataset_name,
        'best_params': best_params,
        'best_score': best_score,
        'baseline_score': best_model_info['val_f1_score'],
        'improvement': best_score - best_model_info['val_f1_score'],
        'train_size': len(train_text),
        'val_size': len(val_text),
        'final_train_size': len(train_val_text)
    }
    
    if _save_tuning_results(results, final_pipeline, output_dir, model_name, dataset_name):
        print(f"\nTuning complete! Results saved to: {output_dir}")
        return results, final_pipeline
    else:
        print("Failed to save tuning results")
        return None

def load_best_tuned_model():
    """Load the best tuned model for final evaluation"""
    best_model_info = get_best_model()
    if not best_model_info:
        print("No model selection results found")
        return None
    
    model_name = best_model_info['model_name']
    dataset_name = best_model_info['dataset']
    
    model_file = f"outputs/hyperparameter_tuning/{model_name}_{dataset_name}_tuned_model.pkl"
    results_file = f"outputs/hyperparameter_tuning/{model_name}_{dataset_name}_best_params.json"
    
    if os.path.exists(model_file) and os.path.exists(results_file):
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return model, results
        except Exception as e:
            print(f"Failed to load tuned model: {str(e)}")
            return None
    else:
        print("No tuned model found. Please run hyperparameter tuning first.")
        return None
