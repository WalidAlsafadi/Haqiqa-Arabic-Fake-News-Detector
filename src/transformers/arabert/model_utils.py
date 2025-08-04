"""
AraBERT model utilities for Palestinian fake news detection.
Provides helper functions for model management.
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple, Optional


def load_arabert_model(model_path: str, device: Optional[str] = None) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load AraBERT model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        device: Device to load model on (auto-detect if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading AraBERT model from: {model_path}")
    print(f"Using device: {device}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


def save_arabert_model(model, tokenizer, output_path: str):
    """
    Save AraBERT model and tokenizer.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        output_path: Path to save the model
    """
    print(f"Saving model to: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Save model
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_path)
        else:
            # If it's a trainer, use the model attribute
            model.model.save_pretrained(output_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        print("Model and tokenizer saved successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {output_path}: {str(e)}")


def get_model_info(model_path: str) -> dict:
    """
    Get information about a saved AraBERT model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary with model information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    info = {
        "model_path": model_path,
        "exists": True,
        "files": []
    }
    
    # Check for required files
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            info["files"].append(file)
            info[f"{file}_exists"] = True
            info[f"{file}_size"] = os.path.getsize(file_path)
        else:
            info[f"{file}_exists"] = False
    
    return info


def verify_model_compatibility(model_path: str) -> bool:
    """
    Verify if the model is compatible and can be loaded.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        True if model can be loaded, False otherwise
    """
    try:
        # Try to load the model configuration
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        
        # Check if it's a classification model with 2 labels
        if hasattr(config, 'num_labels') and config.num_labels == 2:
            print(f"✅ Model is compatible (2 labels for binary classification)")
            return True
        else:
            print(f"❌ Model has {getattr(config, 'num_labels', 'unknown')} labels, expected 2")
            return False
            
    except Exception as e:
        print(f"❌ Model verification failed: {str(e)}")
        return False


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return info


if __name__ == "__main__":
    # Example usage
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Check if model exists
    model_path = "models/arabert/finetuned-model"
    if os.path.exists(model_path):
        model_info = get_model_info(model_path)
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Verify compatibility
        is_compatible = verify_model_compatibility(model_path)
        print(f"\nModel compatibility: {'✅ Compatible' if is_compatible else '❌ Not compatible'}")
    else:
        print(f"\nModel not found at: {model_path}")
