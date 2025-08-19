"""
AraBERT module for Palestinian fake news detection.
"""

from .training import train_arabert_model
from .evaluation import evaluate_arabert_model  
from .model_utils import load_arabert_model, save_arabert_model

__all__ = [
    'train_arabert_model',
    'evaluate_arabert_model', 
    'load_arabert_model',
    'save_arabert_model'
]
