"""
AraBERT training module for Palestinian fake news detection.
Consolidates training logic from colab notebooks.
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
import torch
from datasets import Dataset


def prepare_data(data_path: str, test_size: float = 0.4, random_state: int = 42):
    """
    Prepare and split data for training.
    
    Args:
        data_path: Path to the CSV dataset
        test_size: Fraction for temp split (val + test)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = pd.read_csv(data_path)
    X = df["text"]
    y = df["label"]

    # First split: 60% train, 40% temp (which becomes 20% val + 20% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: Split the 40% temp into 20% val and 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_datasets(X_train, X_val, y_train, y_val, tokenizer, max_length: int = 512):
    """
    Create and tokenize datasets for training.
    
    Args:
        X_train, X_val: Training and validation texts
        y_train, y_val: Training and validation labels
        tokenizer: Pre-trained tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)

    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": X_train.tolist(),
        "labels": y_train.tolist()
    })
    val_dataset = Dataset.from_dict({
        "text": X_val.tolist(),
        "labels": y_val.tolist()
    })

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation during training.
    
    Args:
        eval_pred: Evaluation predictions from trainer
        
    Returns:
        Dictionary of computed metrics
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


def train_arabert_model(
    data_path: str,
    output_dir: str,
    model_name: str = "aubmindlab/bert-base-arabertv02",
    num_epochs: int = 7,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    logging_steps: int = 100,
    save_total_limit: int = 1
):
    """
    Train AraBERT model for fake news detection.
    
    Args:
        data_path: Path to the training dataset
        output_dir: Directory to save the trained model
        model_name: Pre-trained model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps
        logging_steps: Logging frequency
        save_total_limit: Maximum number of checkpoints to save
        
    Returns:
        Trained model and tokenizer
    """
    print("=" * 60)
    print("Palestine Fake News Detection - AraBERT Fine-Tuning")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLOADING ARABERT MODEL AND TOKENIZER")
    print("=" * 60)
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1}
    )
    
    # Initialize weights manually
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    model.classifier.bias.data.zero_()
    print("AraBERT and Tokenizer loaded successfully")
    
    print("\nLOADING AND PREPARING DATA")
    print("=" * 60)
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(X_train, X_val, y_train, y_val, tokenizer)
    print("Data loaded and prepared successfully")
    
    print("\nSTARTING TRAINING")
    print("=" * 60)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    print("Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\nTRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print("Ready for evaluation")
    
    return trainer, tokenizer


if __name__ == "__main__":
    # Example usage
    from src.config.settings import TRANSFORMERS_DATASET_PATH, ARABERT_MODEL_PATH
    
    train_arabert_model(
        data_path=TRANSFORMERS_DATASET_PATH,
        output_dir=ARABERT_MODEL_PATH
    )
