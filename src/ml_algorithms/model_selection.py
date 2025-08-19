"""
Model Selection Module for Palestine Fake News Detection

Compares multiple ML algorithms using cross-validation to identify the best performing model.
Supports both minimal and aggressive text cleaning approaches.

Models tested:
- XGBoost
- Random Forest  
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
"""

import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.config.settings import RANDOM_STATE
from src.config.settings import TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF
from src.utils.data_splits import DataSplitter

def _validate_inputs(splitter, dataset_names):
    """Validate inputs for model selection"""
    if splitter.train_indices is None:
        raise ValueError("DataSplitter has no splits. Call create_splits() first.")
    
    valid_datasets = ['minimal', 'aggressive']
    invalid_datasets = [name for name in dataset_names if name not in valid_datasets]
    if invalid_datasets:
        raise ValueError(f"Invalid dataset names: {invalid_datasets}. Valid options: {valid_datasets}")

def _define_models():
    """Define models for quick screening phase (class imbalance handled in hyperparameter tuning)"""
    return {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, 
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='linear', 
            random_state=RANDOM_STATE
        ),
        'Naive Bayes': MultinomialNB()
    }

def _create_tfidf_pipeline(model):
    """Create TF-IDF pipeline for a model"""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words=None
        )),
        ('classifier', model)
    ])

def _print_ascii_table(results, title):
    """Print results in professional ASCII table format"""
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Model':<20} {'CV F1 Mean':<12} {'CV F1 Std':<12} {'Val F1':<12} {'Rank':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model']:<20} {result['cv_f1_mean']:<12.4f} {result['cv_f1_std']:<12.4f} "
              f"{result['val_f1_score']:<12.4f} {result['rank']:<8}")
    print("-" * 80)

def _perform_cross_validation(X_train, y_train, model_name, pipeline):
    """Perform 5-fold cross-validation on training data"""
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=5, scoring='f1_weighted', 
        n_jobs=-1
    )
    
    cv_f1_mean = cv_scores.mean()
    cv_f1_std = cv_scores.std()
    
    print(f"{model_name:<20} CV F1: {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")
    
    return cv_f1_mean, cv_f1_std

def _evaluate_on_validation(pipeline, X_train, y_train, X_val, y_val, model_name):
    """Train on training data and evaluate on validation data"""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"{model_name:<20} Val F1: {val_f1:.4f}")
    
    return val_f1

def compare_models(splitter, dataset_names, output_dir):
    """
    Compare models using 5-fold cross-validation on training data + validation evaluation.
    
    Methodology:
    1. Use 5-fold CV on training data (2,712 samples) for robust model comparison
    2. Train final models on full training data and evaluate on validation data (904 samples)
    3. Rank models by CV F1 score for hyperparameter tuning selection
    4. Maintain 60/20/20 split integrity throughout process
    """
    _validate_inputs(splitter, dataset_names)
    
    print("\nMODEL SELECTION WITH CROSS-VALIDATION")
    print("=" * 60)
    print("Methodology: 5-fold CV on training data + validation evaluation")
    print(f"Training samples: {len(splitter.train_indices)}")
    print(f"Validation samples: {len(splitter.val_indices)}")
    print(f"Test samples: {len(splitter.test_indices)} (held out)")
    
    all_results = []
    models = _define_models()
    
    for dataset_name in dataset_names:
        print(f"\n\nDATASET: {dataset_name.upper()}")
        print("=" * 40)
        
        # Load data using data splitter methods
        from src.config.settings import DATASET_PATHS
        data_path = DATASET_PATHS.get(dataset_name)
        if not data_path or not os.path.exists(data_path):
            print(f"Skipping {dataset_name}: file not found at {data_path}")
            continue
        
        # Use data splitter methods to get train and validation data
        X_train, y_train = splitter.get_train_from_csv(data_path, text_column='text')
        X_val, y_val = splitter.get_val_from_csv(data_path, text_column='text')
        
        dataset_results = []
        
        print(f"\nPerforming 5-fold cross-validation on training data...")
        print("-" * 60)
        
        for model_name, model in models.items():
            pipeline = _create_tfidf_pipeline(model)
            
            # Perform cross-validation on training data only
            cv_f1_mean, cv_f1_std = _perform_cross_validation(
                X_train, y_train, model_name, pipeline
            )
            
            # Evaluate on validation data
            val_f1 = _evaluate_on_validation(
                pipeline, X_train, y_train, X_val, y_val, model_name
            )
            
            result = {
                'dataset': dataset_name,
                'model': model_name,
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std,
                'val_f1_score': val_f1,
                'pipeline': pipeline  # Store for saving best model
            }
            
            dataset_results.append(result)
        
        # Rank models by CV F1 score (primary metric for model selection)
        dataset_results.sort(key=lambda x: x['cv_f1_mean'], reverse=True)
        for i, result in enumerate(dataset_results, 1):
            result['rank'] = i
        
        # Print results table
        _print_ascii_table(dataset_results, f"RESULTS SUMMARY - {dataset_name.upper()}")
        
        # Save best model pipeline
        best_result = dataset_results[0]
        best_pipeline_path = os.path.join(output_dir, f'best_pipeline_{dataset_name}.pkl')
        with open(best_pipeline_path, 'wb') as f:
            pickle.dump(best_result['pipeline'], f)
        print(f"\nBest model saved: {best_result['model']} (CV F1: {best_result['cv_f1_mean']:.4f})")
        
        # Generate classification report for best model
        best_pipeline = best_result['pipeline']
        y_pred = best_pipeline.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Save detailed results
        report_path = os.path.join(output_dir, f'{best_result["model"]}_{dataset_name}_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {best_result['model']}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"CV F1 Mean: {best_result['cv_f1_mean']:.4f}\n")
            f.write(f"CV F1 Std: {best_result['cv_f1_std']:.4f}\n")
            f.write(f"Validation F1: {best_result['val_f1_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_val, y_pred))
        
        all_results.extend(dataset_results)
    
    # Save comprehensive results
    results_path = os.path.join(output_dir, 'model_selection_results.json')
    serializable_results = []
    for result in all_results:
        serializable_result = {k: v for k, v in result.items() if k != 'pipeline'}
        serializable_results.append(serializable_result)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n\nMODEL SELECTION COMPLETE")
    print("=" * 40)
    print(f"Results saved to: {output_dir}")
    print("Next step: Hyperparameter tuning on best models")
    
    return all_results

# Legacy compatibility functions (maintained for backward compatibility)
def compare_models_on_datasets(splitter, dataset_names=['minimal', 'aggressive']):
    """Legacy wrapper - redirects to new compare_models function"""
    return compare_models(splitter, dataset_names, "outputs/model_selection")

def get_best_model():
    """Get the best model from model selection results"""
    try:
        results_path = "outputs/model_selection/model_selection_results.json"
        if not os.path.exists(results_path):
            return None
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Find best model by CV F1 score
        best_result = max(results, key=lambda x: x['cv_f1_mean'])
        
        return {
            'model_name': best_result['model'],
            'dataset': best_result['dataset'],
            'cv_f1_mean': best_result['cv_f1_mean'],
            'val_f1_score': best_result['val_f1_score']
        }
    except Exception as e:
        print(f"Failed to get best model: {str(e)}")
        return None

def load_best_pipeline():
    """Load the best trained pipeline for hyperparameter tuning"""
    best_model = get_best_model()
    if not best_model:
        print("No best model found")
        return None
    
    pipeline_file = f"outputs/model_selection/best_pipeline_{best_model['dataset']}.pkl"
    
    try:
        if os.path.exists(pipeline_file):
            with open(pipeline_file, 'rb') as f:
                pipeline = pickle.load(f)
            return pipeline, best_model
        else:
            print(f"Best pipeline file not found: {pipeline_file}")
            return None
    except Exception as e:
        print(f"Failed to load pipeline: {str(e)}")
        return None
