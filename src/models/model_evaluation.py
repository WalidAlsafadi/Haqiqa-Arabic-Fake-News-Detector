"""Final model evaluation using test set"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

from src.utils.data_splits import DataSplitter
from src.models.hyperparameter_tuning import load_best_tuned_model

plt.style.use('default')
sns.set_palette("husl")

def _validate_inputs(splitter):
    """Validate inputs for model evaluation"""
    if splitter.train_indices is None:
        raise ValueError("DataSplitter has no splits. Call create_splits() first.")

def _load_tuned_model():
    """Load the best tuned model with error handling"""
    result = load_best_tuned_model()
    if result is None:
        print("No tuned model found!")
        print("Please run hyperparameter tuning first")
        return None
    
    tuned_pipeline, tuning_results = result
    model_name = tuning_results['model_name']
    dataset_name = tuning_results['dataset']
    
    print(f"Evaluating: {model_name} on {dataset_name} dataset")
    print(f"Tuned validation F1: {tuning_results['best_score']:.4f}")
    
    return tuned_pipeline, tuning_results

def _calculate_metrics(test_labels, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(test_labels, y_pred),
        'f1_weighted': f1_score(test_labels, y_pred, average='weighted'),
        'f1_macro': f1_score(test_labels, y_pred, average='macro'),
        'precision': precision_score(test_labels, y_pred, average='weighted'),
        'recall': recall_score(test_labels, y_pred, average='weighted')
    }
    
    # Class-specific metrics (assuming binary: 0=Real, 1=Fake)
    f1_per_class = f1_score(test_labels, y_pred, average=None)
    metrics['fake_news_f1'] = f1_per_class[1] if len(f1_per_class) > 1 else metrics['f1_weighted']
    metrics['real_news_f1'] = f1_per_class[0] if len(f1_per_class) > 1 else metrics['f1_weighted']
    
    # AUC if probabilities available
    metrics['auc_score'] = None
    if y_pred_proba is not None:
        try:
            metrics['auc_score'] = roc_auc_score(test_labels, y_pred_proba)
        except Exception as e:
            print(f"Could not calculate AUC: {str(e)}")
            metrics['auc_score'] = None
    
    return metrics

def _save_detailed_results(metrics, model_name, dataset_name, test_text, test_labels, y_pred, tuning_results, output_dir):
    """Save detailed evaluation results to file"""
    try:
        results_file = f"{output_dir}/{model_name}_{dataset_name}_results.txt"
        
        # Generate classification report and confusion matrix
        class_report = classification_report(test_labels, y_pred, digits=4)
        conf_matrix = confusion_matrix(test_labels, y_pred)
        
        with open(results_file, 'w') as f:
            f.write("FINAL MODEL EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Test Set Size: {len(test_text)} samples\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now()}\n")
            f.write("\n")
            f.write("SUMMARY METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"Test F1 (macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"Test Precision: {metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {metrics['recall']:.4f}\n")
            f.write(f"Real News F1: {metrics['real_news_f1']:.4f}\n")
            f.write(f"Fake News F1: {metrics['fake_news_f1']:.4f}\n")
            if metrics['auc_score']:
                f.write(f"Test AUC: {metrics['auc_score']:.4f}\n")
            f.write("\n")
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 60 + "\n")
            f.write(class_report)
            f.write("\n\n")
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 30 + "\n")
            f.write(str(conf_matrix))
            f.write("\n")
        
        print(f"Detailed results saved to: {results_file}")
        return conf_matrix
    except Exception as e:
        print(f"Failed to save detailed results: {str(e)}")
        return None

def _generate_evaluation_plots(test_labels, y_pred, y_pred_proba, model_name, dataset_name, conf_matrix, output_dir):
    """Generate evaluation plots"""
    try:
        print("Generating evaluation plots...")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_{dataset_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random (AUC = 0.5)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {model_name} on {dataset_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_{dataset_name}_roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Precision-Recall Curve  
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            precision_vals, recall_vals, _ = precision_recall_curve(test_labels, y_pred_proba)
            avg_precision = average_precision_score(test_labels, y_pred_proba)
            
            # Plot the PR curve
            plt.plot(recall_vals, precision_vals, linewidth=2, label=f'PR curve (AP = {avg_precision:.3f})')
            
            # Add baseline (random classifier)
            pos_ratio = sum(test_labels) / len(test_labels)
            plt.axhline(y=pos_ratio, color='red', linestyle='--', alpha=0.7, label=f'Random (AP = {pos_ratio:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve: {model_name} on {dataset_name}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_{dataset_name}_pr_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"Evaluation plots saved to: {output_dir}")
            
    except Exception as e:
        print(f"Failed to generate plots: {str(e)}")

def _save_deployment_model(tuned_pipeline, model_name):
    """Save final model for deployment"""
    try:
        print("Saving final model for deployment...")
        
        os.makedirs("models/trained", exist_ok=True)
        model_file = "models/trained/best_model.pkl"
        
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(tuned_pipeline, f)
        
        print(f"Final pipeline saved to: {model_file}")
        print("Model is ready for deployment!")
        return True
    except Exception as e:
        print(f"Failed to save deployment model: {str(e)}")
        return False

def evaluate_final_model(splitter, output_dir="outputs/final_evaluation"):
    """
    Final evaluation using held-out test set.
    
    ML Best Practices:
    1. Uses ONLY the test set that was held out from the beginning
    2. Test set has never been seen during training or tuning
    3. Uses the best tuned pipeline from hyperparameter tuning
    
    Args:
        splitter: DataSplitter instance with loaded splits
        output_dir: Directory to save plots and reports
    
    Returns:
        dict: Final test metrics for reporting
    """
    
    # Validate inputs
    _validate_inputs(splitter)
    
    print("\nFINAL MODEL EVALUATION")
    print("=" * 60)
    print("Using held-out test set for final evaluation")
    print("Test set has never been used during training or tuning")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the best tuned model
    result = _load_tuned_model()
    if result is None:
        return None
    
    tuned_pipeline, tuning_results = result
    model_name = tuning_results['model_name']
    dataset_name = tuning_results['dataset']
    
    # Dataset file mapping
    dataset_files = {
        'minimal': 'data/processed/minimal_cleaned.csv',
        'aggressive': 'data/processed/aggressive_cleaned.csv'
    }
    
    # Get the held-out test set from CSV
    csv_path = dataset_files[dataset_name]
    if not os.path.exists(csv_path):
        print(f"Dataset file not found: {csv_path}")
        return None
        
    test_text, test_labels = splitter.get_test_from_csv(csv_path)
    print(f"Test set size: {len(test_text)} samples")
    
    # Make predictions using the complete pipeline
    y_pred = tuned_pipeline.predict(test_text.fillna(''))
    
    # Get prediction probabilities if available
    y_pred_proba = None
    try:
        if hasattr(tuned_pipeline, "predict_proba"):
            proba_matrix = tuned_pipeline.predict_proba(test_text.fillna(''))
            y_pred_proba = proba_matrix[:, 1]  # Probability of positive class (fake news)
        elif hasattr(tuned_pipeline, "decision_function"):
            # For SVM, decision function gives signed distance to hyperplane
            y_pred_proba = tuned_pipeline.decision_function(test_text.fillna(''))
        else:
            print("Model doesn't support probability predictions")
    except Exception as e:
        print(f"Could not get prediction probabilities: {str(e)}")
        y_pred_proba = None
    
    # Calculate comprehensive metrics
    metrics = _calculate_metrics(test_labels, y_pred, y_pred_proba)
    
    # Display results in professional ASCII table format
    print("\nFINAL TEST RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':<12}")
    print("-" * 35)
    print(f"{'Test Accuracy':<20} {metrics['accuracy']:<12.4f}")
    print(f"{'Test F1 (weighted)':<20} {metrics['f1_weighted']:<12.4f}")
    print(f"{'Test F1 (macro)':<20} {metrics['f1_macro']:<12.4f}")
    print(f"{'Test Precision':<20} {metrics['precision']:<12.4f}")
    print(f"{'Test Recall':<20} {metrics['recall']:<12.4f}")
    print(f"{'Real News F1':<20} {metrics['real_news_f1']:<12.4f}")
    print(f"{'Fake News F1':<20} {metrics['fake_news_f1']:<12.4f}")
    if metrics['auc_score']:
        print(f"{'Test AUC':<20} {metrics['auc_score']:<12.4f}")
    print("-" * 35)
    
    # Save detailed results
    conf_matrix = _save_detailed_results(
        metrics, model_name, dataset_name, test_text, test_labels, y_pred, tuning_results, output_dir
    )
    
    if conf_matrix is not None:
        # Generate evaluation plots
        _generate_evaluation_plots(test_labels, y_pred, y_pred_proba, model_name, dataset_name, conf_matrix, output_dir)
        
        # Save final model for deployment
        _save_deployment_model(tuned_pipeline, model_name)
    
    # Return comprehensive results
    results = {
        'model_name': model_name,
        'dataset': dataset_name,
        'test_accuracy': metrics['accuracy'],
        'test_f1_weighted': metrics['f1_weighted'],
        'test_f1_macro': metrics['f1_macro'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'fake_news_f1': metrics['fake_news_f1'],
        'real_news_f1': metrics['real_news_f1'],
        'test_auc': metrics['auc_score'],
        'test_size': len(test_text),
        'confusion_matrix': conf_matrix.tolist() if conf_matrix is not None else None,
        'tuning_validation_f1': tuning_results['best_score'],
        'baseline_f1': tuning_results['baseline_score']
    }
    
    print("=" * 60)
    return results

if __name__ == "__main__":
    print("Starting final model evaluation...")
    
    # Create data splitter instance
    from src.utils.data_splits import DataSplitter
    splitter = DataSplitter()
    
    # Load existing splits
    try:
        splitter = DataSplitter.load_splits()
        print("Loaded existing data splits")
    except Exception as e:
        print(f"Failed to load splits: {str(e)}")
        print("Creating new data splits...")
        
        try:
            from src.preprocessing.text_cleaner import prepare_data
            from src.config.settings import DATASET_PATH
            
            # Prepare data with all cleaning approaches
            df = prepare_data(DATASET_PATH)
            
            # Create splits
            splitter.create_splits(df)
            print("Data splits created and saved")
        except Exception as e:
            print(f"Failed to create splits: {str(e)}")
            exit(1)
    
    # Run final evaluation
    results = evaluate_final_model(splitter)
    
    if results:
        print("\nFinal evaluation completed successfully!")
        print(f"Best model: {results['model_name']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Final test F1: {results['test_f1_weighted']:.4f}")
        
        # Compare with validation performance
        val_f1 = results['tuning_validation_f1']
        test_f1 = results['test_f1_weighted']
        performance_drop = val_f1 - test_f1
        
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Performance drop: {performance_drop:.4f}")
        
        if performance_drop > 0.05:
            print("Significant performance drop detected!")
            print("This might indicate overfitting to the validation set")
        else:
            print("Good generalization to test set")
    else:
        print("Final evaluation failed!")
        print("Please ensure hyperparameter tuning has been completed")
