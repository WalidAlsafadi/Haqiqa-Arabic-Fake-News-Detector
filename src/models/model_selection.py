"""Model selection with clean output and consistent evaluation"""

import os
import json
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import pandas as pd

from src.config.settings import CV_FOLDS, RANDOM_STATE, TEST_SIZE

def save_model_selection_results(results, output_dir="outputs/model_selection"):
    """Save model selection results to file"""
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/model_selection_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for key, result in results.items():
        json_results[key] = {
            'model_name': result['model_name'],
            'dataset': result['dataset'],
            'cv_f1_mean': float(result['cv_f1_mean']),
            'cv_f1_std': float(result['cv_f1_std']),
            'report_file': result.get('report_file', '')
        }
        
        # Save vectorizer as pickle file for hyperparameter tuning
        if 'vectorizer' in result:
            vectorizer_file = f"{output_dir}/{key}_vectorizer.pkl"
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(result['vectorizer'], f)
            print(f"  Saved vectorizer: {vectorizer_file}")
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Model selection results saved to {results_file}")
    return results_file

def load_model_selection_results(output_dir="outputs/model_selection"):
    """Load previously saved model selection results"""
    results_file = f"{output_dir}/model_selection_results.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def compare_models_on_datasets(splits, dataset_names=['minimal', 'aggressive']):
    """
    Compare models on both minimal and aggressive datasets using proper train/validation splits.
    
    ML Best Practices:
    1. Uses only train+validation data (test set is held out)
    2. Fits TF-IDF only on training data
    3. Uses cross-validation on train+validation combined
    4. Saves fitted vectorizers for consistent use in later phases
    
    Args:
        splits: Data splits from create_data_splits()
        dataset_names: List of dataset names to compare
    
    Returns:
        dict: Results for each model-dataset combination
    """
    
    from src.config.settings import TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF
    from src.utils.data_splits import get_train_data, get_train_val_data
    
    # Simple parameters for model selection phase
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=500
        ),
        'SVM': SVC(
            random_state=RANDOM_STATE
        ),
        'Naive Bayes': MultinomialNB()
    }
    
    print("=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    print("Strategy: Cross-validation on train+validation data only")
    print("Test set is held out for final evaluation")
    print()
    
    all_results = {}
    
    for dataset_name in dataset_names:
        print(f"Dataset: {dataset_name}")
        print("-" * 40)
        
        # Get train data for TF-IDF fitting
        train_text, train_labels = get_train_data(splits, dataset_name)
        
        # Get train+validation data for cross-validation
        train_val_text, train_val_labels = get_train_val_data(splits, dataset_name)
        
        print(f"Train data for TF-IDF fitting: {len(train_text)} samples")
        print(f"Train+Val data for CV: {len(train_val_text)} samples")
        
        # Create TF-IDF vectorizer and fit ONLY on training data
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words=None  # We handle Arabic stopwords in preprocessing
        )
        
        # Fit vectorizer ONLY on training text (proper ML practice)
        print("Fitting TF-IDF vectorizer on training data only...")
        vectorizer.fit(train_text.fillna(''))
        
        for model_name, model in models.items():
            try:
                # For cross-validation, we need to create a pipeline to ensure proper fitting
                from sklearn.pipeline import Pipeline
                
                # Create a fresh vectorizer for the pipeline (will be fitted during CV)
                cv_vectorizer = TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    min_df=TFIDF_MIN_DF,
                    max_df=TFIDF_MAX_DF,
                    stop_words=None
                )
                
                cv_pipeline = Pipeline([
                    ('tfidf', cv_vectorizer),
                    ('classifier', model)
                ])
                
                # Cross-validation with proper pipeline (prevents data leakage)
                cv_scores = cross_val_score(
                    cv_pipeline, 
                    train_val_text.fillna(''), 
                    train_val_labels, 
                    cv=CV_FOLDS, 
                    scoring='f1_weighted'
                )
                cv_f1_mean = cv_scores.mean()
                cv_f1_std = cv_scores.std()
                
                # Train final model for classification report (using the fitted vectorizer)
                X_train = vectorizer.transform(train_text.fillna(''))
                model.fit(X_train, train_labels)
                
                # Create a small validation split for classification report
                from sklearn.model_selection import train_test_split
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, train_labels, test_size=0.25, random_state=RANDOM_STATE, stratify=train_labels
                )
                
                # Retrain on the smaller training split
                model.fit(X_train_split, y_train_split)
                y_pred = model.predict(X_val_split)
                
                # Generate classification report as text
                class_report_text = classification_report(y_val_split, y_pred, digits=4)
                
                # Save classification report as .txt file
                os.makedirs("outputs/model_selection", exist_ok=True)
                report_filename = f"outputs/model_selection/{model_name}_{dataset_name}_report.txt"
                
                with open(report_filename, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Cross-Validation F1 (5-fold): {cv_f1_mean:.4f} (+/- {cv_f1_std*2:.4f})\n")
                    f.write("=" * 60 + "\n")
                    f.write("CLASSIFICATION REPORT (Train/Val Split within Training Data):\n")
                    f.write("=" * 60 + "\n")
                    f.write(class_report_text)
                    f.write("\n" + "=" * 60 + "\n")
                    f.write(f"Note: CV F1 score ({cv_f1_mean:.4f}) is used for model comparison\n")
                    f.write("Note: Test set is held out and not used in model selection\n")
                
                result_key = f"{model_name}_{dataset_name}"
                all_results[result_key] = {
                    'model_name': model_name,
                    'dataset': dataset_name,
                    'cv_f1_mean': cv_f1_mean,
                    'cv_f1_std': cv_f1_std,
                    'model': model,
                    'vectorizer': vectorizer,  # Save the fitted vectorizer
                    'report_file': report_filename
                }
                
                print(f"  {model_name:<20} CV F1: {cv_f1_mean:.4f} (+/- {cv_f1_std*2:.4f})")
                
            except Exception as e:
                print(f"  {model_name:<20} FAILED: {str(e)}")
                continue
        
        print()
    
    # Find best 2 models overall (Option A)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)
    
    print("=" * 60)
    print("MODEL RANKING (by Cross-Validation F1 Score)")
    print("=" * 60)
    for i, (key, result) in enumerate(sorted_results):
        print(f"{i+1:2d}. {result['model_name']:<20} ({result['dataset']:<10}) F1: {result['cv_f1_mean']:.4f}")
    
    best_2 = sorted_results[:2]
    print()
    print("SELECTED FOR HYPERPARAMETER TUNING:")
    for i, (key, result) in enumerate(best_2):
        print(f"  {i+1}. {result['model_name']} on {result['dataset']} dataset (F1: {result['cv_f1_mean']:.4f})")
    
    print("=" * 60)
    
    # Save results
    save_model_selection_results(all_results)
    
    return all_results

def compare_models(X, y):
    """Backward compatibility function for old imports"""
    # Simple model comparison for backwards compatibility
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=500),
        'SVM': SVC(random_state=RANDOM_STATE),
        'Naive Bayes': MultinomialNB()
    }
    
    results = {}
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='f1_weighted')
            results[name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'model': model
            }
        except:
            continue
    
    # Save results for future use
    save_model_selection_results(results)
    
    return results

def evaluate_best_model(results, X, y):
    """Evaluate the best model with train/val/test split"""
    
    # Find best model by F1 score
    best_name = max(results.keys(), key=lambda k: results[k]['mean_f1'])
    best_model = results[best_name]['model']
    
    print(f"Best model: {best_name}")
    
    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]} samples")
    
    # Train on training set
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    return best_model, {
        'name': best_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

def get_vectorizers_from_model_selection():
    """Get fitted vectorizers from model selection results"""
    import pickle
    
    results = load_model_selection_results()
    if not results:
        return None
    
    vectorizers = {}
    # The saved results only have JSON-serializable data
    # We need to load vectorizers separately from their pickle files
    for key, result in results.items():
        vectorizer_file = f"outputs/model_selection/{key}_vectorizer.pkl"
        if os.path.exists(vectorizer_file):
            with open(vectorizer_file, 'rb') as f:
                vectorizers[key] = pickle.load(f)
    
    return vectorizers

def get_best_models_with_vectorizers():
    """Get the best 2 models with their corresponding vectorizers"""
    import pickle
    
    # Load model selection results
    results = load_model_selection_results()
    if not results:
        return None, None
    
    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)
    best_2 = sorted_results[:2]
    
    models = {}
    vectorizers = {}
    
    for key, result in best_2:
        # Load model
        model_file = f"outputs/model_selection/{key}_model.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                models[key] = pickle.load(f)
        
        # Load vectorizer  
        vectorizer_file = f"outputs/model_selection/{key}_vectorizer.pkl"
        if os.path.exists(vectorizer_file):
            with open(vectorizer_file, 'rb') as f:
                vectorizers[key] = pickle.load(f)
    
    return models, vectorizers
