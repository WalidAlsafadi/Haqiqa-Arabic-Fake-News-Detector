"""
AraBERT evaluation module for Palestinian fake news detection.
Consolidates evaluation logic from colab notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax


def load_test_data(data_path: str, random_state: int = 42):
    """
    Load and prepare test data using the same splits as training.
    
    Args:
        data_path: Path to the dataset
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, true_labels)
    """
    df = pd.read_csv(data_path)
    X = df["text"]
    y = df["label"]

    # Use SAME data splits as training (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )
    
    return X_test.tolist(), y_test.tolist()


def predict_batch(model, tokenizer, texts, device, batch_size: int = 32):
    """
    Make predictions on a batch of texts.
    
    Args:
        model: Trained AraBERT model
        tokenizer: Model tokenizer
        texts: List of texts to predict
        device: Computing device
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    predictions = []
    probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Convert to probabilities
        probs = softmax(logits.cpu().numpy(), axis=1)
        
        # Get predictions and probabilities
        batch_preds = (probs[:, 1] > probs[:, 0]).astype(int)
        batch_probs = probs[:, 1]  # Probability of fake
        
        predictions.extend(batch_preds.tolist())
        probabilities.extend(batch_probs.tolist())
        
        # Progress tracking
        if (i // batch_size + 1) % 10 == 0:
            print(f"Progress: {i + len(batch_texts)}/{len(texts)} ({(i + len(batch_texts))/len(texts)*100:.1f}%)")
    
    return predictions, probabilities


def generate_plots(true_labels, predictions, probabilities, output_dir: str):
    """
    Generate evaluation plots.
    
    Args:
        true_labels: True labels
        predictions: Model predictions
        probabilities: Prediction probabilities
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("AraBERT Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AraBERT_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AraBERT (AUC = {roc_auc:.3f})", linewidth=3)
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray', alpha=0.7)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("AraBERT ROC Curve", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AraBERT_roc_curve.png"), dpi=300)
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    pr_precision, pr_recall, _ = precision_recall_curve(true_labels, probabilities)
    plt.plot(pr_recall, pr_precision, label=f"AraBERT", linewidth=3)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("AraBERT Precision-Recall Curve", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AraBERT_pr_curve.png"), dpi=300)
    plt.close()
    
    return roc_auc


def save_results(true_labels, predictions, probabilities, output_dir: str):
    """
    Save evaluation results to text file.
    
    Args:
        true_labels: True labels
        predictions: Model predictions
        probabilities: Prediction probabilities
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    precision_val = precision_score(true_labels, predictions)
    recall_val = recall_score(true_labels, predictions)
    
    # Classification report
    report = classification_report(true_labels, predictions, target_names=['Real', 'Fake'], digits=4)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Save to file
    with open(os.path.join(output_dir, "AraBERT_results.txt"), "w", encoding='utf-8') as f:
        f.write("FINAL MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Model: AraBERT\n")
        f.write("Dataset: transformers\n")
        f.write(f"Test Set Size: {len(true_labels)} samples\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now()}\n\n")

        f.write("SUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1 (weighted): {f1_weighted:.4f}\n")
        f.write(f"Test F1 (macro): {f1_macro:.4f}\n")
        f.write(f"Test Precision: {precision_val:.4f}\n")
        f.write(f"Test Recall: {recall_val:.4f}\n")
        f.write(f"Real News F1: {report.splitlines()[2].split()[3]}\n")
        f.write(f"Fake News F1: {report.splitlines()[3].split()[3]}\n")
        f.write(f"Test AUC: {roc_auc:.4f}\n\n")

        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 60 + "\n")
        f.write(report + "\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        f.write(str(cm) + "\n")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision_val,
        'recall': recall_val,
        'auc': roc_auc
    }


def evaluate_arabert_model(
    model_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 32
):
    """
    Evaluate trained AraBERT model.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the dataset
        output_dir: Directory to save evaluation results
        batch_size: Batch size for inference
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("Palestine Fake News Detection - AraBERT Evaluation")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading AraBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load test data
    print("Loading test data...")
    texts, true_labels = load_test_data(data_path)
    print(f"Test samples: {len(texts)}")
    
    # Make predictions
    print("Evaluating samples...")
    predictions, probabilities = predict_batch(model, tokenizer, texts, device, batch_size)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Generate plots
    print("Generating plots...")
    roc_auc = generate_plots(true_labels, predictions, probabilities, output_dir)
    
    # Save results
    print("Saving results...")
    metrics = save_results(true_labels, predictions, probabilities, output_dir)
    
    print("=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Final AUC Score: {roc_auc:.3f}")
    print(f"Final Accuracy: {accuracy:.3f}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    evaluate_arabert_model(
        model_path="models/arabert/finetuned-model",
        data_path="data/processed/transformers_cleaned.csv",
        output_dir="outputs/arabert_evaluation"
    )
