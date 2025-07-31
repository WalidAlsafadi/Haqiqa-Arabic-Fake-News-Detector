"""Comprehensive model evaluation with visualizations and insights"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
import json

from src.config.settings import RANDOM_STATE, TEST_SIZE

plt.style.use('default')
sns.set_palette("husl")

def evaluate_final_model_properly(tuned_model, splits, dataset_name, model_name, output_dir="outputs/final_evaluation"):
    """
    Proper final evaluation using ONLY the held-out test set.
    This should be used ONLY ONCE at the very end for reporting final results.
    
    ML Best Practices:
    1. Uses ONLY the test set that was held out from the beginning
    2. Test set has never been seen by the model during training or tuning
    3. This gives an unbiased estimate of model performance
    
    Args:
        tuned_model: Best trained and tuned model
        splits: Data splits from create_data_splits()
        dataset_name: Name of the dataset used ('minimal' or 'aggressive')
        model_name: Name of the model
        output_dir: Directory to save plots and reports
    
    Returns:
        dict: Final test metrics for reporting
    """
    
    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    print("Using held-out test set for final evaluation...")
    print("Test set has never been used during training or tuning")
    os.makedirs(output_dir, exist_ok=True)
    
    from src.utils.data_splits import get_test_data
    from src.models.model_selection import get_vectorizers_from_model_selection
    
    # Get the held-out test set
    test_text, test_labels = get_test_data(splits, dataset_name)
    print(f"Test set size: {len(test_text)} samples")
    
    # Get the vectorizer used for this model-dataset combination
    vectorizers = get_vectorizers_from_model_selection()
    vectorizer_key = f"{model_name}_{dataset_name}"
    
    if vectorizer_key not in vectorizers:
        print(f"❌ ERROR: No vectorizer found for {vectorizer_key}")
        return None
    
    vectorizer = vectorizers[vectorizer_key]
    print(f"✅ Using vectorizer: {vectorizer_key}")
    
    # Transform test set using fitted vectorizer (no refitting!)
    X_test = vectorizer.transform(test_text.fillna(''))
    print(f"Test features shape: {X_test.shape}")
    
    # FINAL predictions on held-out test set
    y_pred = tuned_model.predict(X_test)
    
    # Get prediction probabilities if available
    y_pred_proba = None
    if hasattr(tuned_model, "predict_proba"):
        y_pred_proba = tuned_model.predict_proba(X_test)[:, 1]
    elif hasattr(tuned_model, "decision_function"):
        y_pred_proba = tuned_model.decision_function(X_test)
    
    # Calculate comprehensive metrics
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    
    accuracy = accuracy_score(test_labels, y_pred)
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    precision = precision_score(test_labels, y_pred, average='weighted')
    recall = recall_score(test_labels, y_pred, average='weighted')
    
    # Class-specific metrics
    f1_per_class = f1_score(test_labels, y_pred, average=None)
    
    # Assume binary classification: 0=Real, 1=Fake
    fake_news_f1 = f1_per_class[1] if len(f1_per_class) > 1 else f1_weighted
    real_news_f1 = f1_per_class[0] if len(f1_per_class) > 1 else f1_weighted
    
    # AUC if probabilities available
    auc_score = None
    if y_pred_proba is not None:
        try:
            auc_score = roc_auc_score(test_labels, y_pred_proba)
        except:
            auc_score = None
    
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 (weighted): {f1_weighted:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Real News F1: {real_news_f1:.4f}")
    print(f"Fake News F1: {fake_news_f1:.4f}")
    if auc_score:
        print(f"Test AUC: {auc_score:.4f}")
    print("=" * 60)
    
    # Detailed classification report
    class_report = classification_report(test_labels, y_pred, digits=4)
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    # Save detailed results
    results_file = f"{output_dir}/{model_name}_{dataset_name}_final_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("FINAL MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Test Set Size: {len(test_text)} samples\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n")
        f.write("\n")
        f.write("IMPORTANT NOTES:\n")
        f.write("- Test set was held out from the beginning of the pipeline\n")
        f.write("- Test set was NEVER used during model selection or hyperparameter tuning\n")
        f.write("- These results represent unbiased model performance\n")
        f.write("\n")
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1 (weighted): {f1_weighted:.4f}\n")
        f.write(f"Test F1 (macro): {f1_macro:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Real News F1: {real_news_f1:.4f}\n")
        f.write(f"Fake News F1: {fake_news_f1:.4f}\n")
        if auc_score:
            f.write(f"Test AUC: {auc_score:.4f}\n")
        f.write("\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 60 + "\n")
        f.write(class_report)
        f.write("\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        f.write(str(conf_matrix))
        f.write("\n")
    
    print(f"✅ Detailed results saved to {results_file}")
    
    # Store results for later return
    results = {
        'test_accuracy': accuracy,
        'test_f1_weighted': f1_weighted,
        'test_f1_macro': f1_macro,
        'test_precision': precision,
        'test_recall': recall,
        'fake_news_f1': fake_news_f1,
        'real_news_f1': real_news_f1,
        'test_auc': auc_score,
        'test_size': len(test_text),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    # Generate plots (don't return early!)
    print("Generating evaluation plots...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrix saved")
    
    # 2. ROC Curve (if probabilities available)
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name} on {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ ROC curve saved")
    
    # 3. Precision-Recall Curve  
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, y_pred_proba)
        avg_precision = average_precision_score(test_labels, y_pred_proba)
        plt.plot(recall_vals, precision_vals, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {model_name} on {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Precision-Recall curve saved")
    
    # Save final model and vectorizer for Streamlit deployment
    print("Saving final model for Streamlit deployment...")
    import pickle
    
    # Create models/trained directory
    os.makedirs("models/trained", exist_ok=True)
    
    # Save the final model
    model_file = f"models/trained/{model_name}_{dataset_name}_best_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(tuned_model, f)
    print(f"  ✅ Model saved: {model_file}")
    
    # Save the vectorizer 
    vectorizer_file = f"models/trained/fitted_vectorizer.pkl"
    from src.models.model_selection import get_vectorizers_from_model_selection
    vectorizers = get_vectorizers_from_model_selection()
    vectorizer_key = f"{model_name}_{dataset_name}"
    
    if vectorizer_key in vectorizers:
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizers[vectorizer_key], f)
        print(f"  ✅ Vectorizer saved: {vectorizer_file}")
    else:
        print(f"  ⚠️  Vectorizer not found for {vectorizer_key}")
    
    print("✅ Final model deployment files ready!")
    
    return results
    
    # Generate comprehensive final report
    final_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Add metadata for final results
    final_report['model_info'] = {
        'model_name': model_name,
        'dataset_used': dataset_name,
        'test_set_size': int(X_test.shape[0]),
        'train_set_size': int(X_train.shape[0]),
        'note': 'FINAL RESULTS - Use these metrics for main reporting'
    }
    
    # Save FINAL classification report
    final_report_file = f"{output_dir}/FINAL_{model_name}_{dataset_name}_test_results.json"
    with open(final_report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Create separate visualizations instead of subplots
    print("Generating evaluation plots...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve (if probabilities available)
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name} on {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("  ROC curve not available (no probability estimates)")
    
    # 3. Precision-Recall Curve
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {model_name} on {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("  PR curve not available (no probability estimates)")
    
    # 4. Class Distribution
    plt.figure(figsize=(8, 6))
    class_counts = pd.Series(y_test).value_counts().sort_index()
    plt.bar(class_counts.index, class_counts.values, alpha=0.7)
    plt.title(f'Test Set Class Distribution: {model_name} on {dataset_name}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/FINAL_{model_name}_{dataset_name}_class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Extract key metrics for reporting
    final_metrics = {
        'model_name': model_name,
        'dataset': dataset_name,
        'test_accuracy': final_report['accuracy'],
        'test_f1_weighted': final_report['weighted avg']['f1-score'],
        'test_f1_macro': final_report['macro avg']['f1-score'],
        'test_precision_weighted': final_report['weighted avg']['precision'],
        'test_recall_weighted': final_report['weighted avg']['recall'],
        'fake_news_f1': final_report['1']['f1-score'],
        'real_news_f1': final_report['0']['f1-score'],
        'test_set_size': X_test.shape[0]
    }
    
    print()
    print("FINAL TEST RESULTS (for main reporting):")
    print(f"  Model: {model_name} on {dataset_name}")
    print(f"  Test Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"  Test F1 (weighted): {final_metrics['test_f1_weighted']:.4f}")
    print(f"  Test F1 (macro): {final_metrics['test_f1_macro']:.4f}")
    print(f"  Fake News F1: {final_metrics['fake_news_f1']:.4f}")
    print(f"  Real News F1: {final_metrics['real_news_f1']:.4f}")
    print("=" * 60)
    print()
    
    return final_metrics

def evaluate_final_model(model, X, y, model_name, dataset_name, output_dir="outputs/model_evaluation"):
    """
    Comprehensive evaluation of the final best model
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels
        model_name: Name of the model
        dataset_name: Name of the dataset used
        output_dir: Directory to save plots and reports
    """
    
    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Fit model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} Evaluation on {dataset_name} Dataset', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
    else:
        axes[0,1].text(0.5, 0.5, 'ROC Curve not available\n(no predict_proba)', 
                       ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('ROC Curve')
    
    # 3. Precision-Recall Curve
    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        axes[1,0].plot(recall, precision, color='blue', lw=2,
                       label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend(loc="lower left")
    else:
        axes[1,0].text(0.5, 0.5, 'PR Curve not available\n(no predict_proba)', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Precision-Recall Curve')
    
    # 4. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        # Show top 20 features
        n_features = min(20, len(model.feature_importances_))
        indices = np.argsort(model.feature_importances_)[-n_features:]
        
        axes[1,1].barh(range(n_features), model.feature_importances_[indices])
        axes[1,1].set_yticks(range(n_features))
        axes[1,1].set_yticklabels([f'Feature {i}' for i in indices])
        axes[1,1].set_xlabel('Importance')
        axes[1,1].set_title('Top Feature Importances')
    elif hasattr(model, 'coef_'):
        # For linear models, show coefficient magnitudes
        coef_abs = np.abs(model.coef_[0])
        n_features = min(20, len(coef_abs))
        indices = np.argsort(coef_abs)[-n_features:]
        
        axes[1,1].barh(range(n_features), coef_abs[indices])
        axes[1,1].set_yticks(range(n_features))
        axes[1,1].set_yticklabels([f'Feature {i}' for i in indices])
        axes[1,1].set_xlabel('|Coefficient|')
        axes[1,1].set_title('Top Feature Coefficients')
    else:
        axes[1,1].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Feature Importance')
    
    plt.tight_layout()
    plot_file = f"{output_dir}/{model_name}_{dataset_name}_evaluation_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Error analysis
    errors = X_test[y_test != y_pred]
    error_analysis = {
        'total_errors': len(errors),
        'error_rate': len(errors) / len(y_test),
        'false_positives': len(y_test[(y_test == 0) & (y_pred == 1)]),
        'false_negatives': len(y_test[(y_test == 1) & (y_pred == 0)])
    }
    
    # Save comprehensive results
    results = {
        'model_name': model_name,
        'dataset': dataset_name,
        'test_size': len(y_test),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'error_analysis': error_analysis,
        'evaluation_plots': plot_file
    }
    
    if y_pred_proba is not None:
        results['roc_auc'] = float(roc_auc)
        results['average_precision'] = float(avg_precision)
    
    # Save results to JSON
    results_file = f"{output_dir}/{model_name}_{dataset_name}_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create detailed text report
    report_file = f"{output_dir}/{model_name}_{dataset_name}_detailed_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"DETAILED EVALUATION REPORT\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Test samples: {len(y_test)}\n\n")
        
        f.write("CLASSIFICATION METRICS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Accuracy: {class_report['accuracy']:.4f}\n")
        f.write(f"Macro F1: {class_report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted F1: {class_report['weighted avg']['f1-score']:.4f}\n\n")
        
        if y_pred_proba is not None:
            f.write("PROBABILITY-BASED METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"Average Precision: {avg_precision:.4f}\n\n")
        
        f.write("ERROR ANALYSIS:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total errors: {error_analysis['total_errors']}\n")
        f.write(f"Error rate: {error_analysis['error_rate']:.4f}\n")
        f.write(f"False positives: {error_analysis['false_positives']}\n")
        f.write(f"False negatives: {error_analysis['false_negatives']}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 35 + "\n")
        f.write(classification_report(y_test, y_pred))
    
    print(f"Evaluation completed for {model_name}")
    print(f"Plots saved to: {plot_file}")
    print(f"Results saved to: {results_file}")
    print(f"Detailed report: {report_file}")
    print("=" * 60)
    
    return results
