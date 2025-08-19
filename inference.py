"""
Efficient inference for Palestine Fake News Detection
Choose between ML (XGBoost winner) or AraBERT models.
Models are loaded only when selected for better performance.
"""

import os
import sys
import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config.settings import BEST_TRADITIONAL_MODEL_PATH, ARABERT_MODEL_PATH
from src.preprocessing.data_preparation import clean_arabic_text_minimal, clean_arabic_text_transformers

class MLPredictor:
    """XGBoost model predictor using minimal cleaning (winner approach)."""
    def __init__(self):
        self.model = None
        
    def load_model(self):
        """Load ML model when needed."""
        if self.model is None:
            print("🔄 Loading XGBoost model...")
            with open(BEST_TRADITIONAL_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ XGBoost model loaded!")
    
    def predict(self, text):
        """Predict using minimal cleaning approach (winner model)."""
        cleaned = clean_arabic_text_minimal(text)
        pred = self.model.predict([cleaned])[0]
        probs = self.model.predict_proba([cleaned])[0]
        predicted_class = "Fake" if pred == 1 else "Real"
        confidence = float(max(probs))
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {"real": float(probs[0]), "fake": float(probs[1])},
            "cleaned_text": cleaned
        }

class AraBERTPredictor:
    """AraBERT model predictor."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load AraBERT model when needed."""
        if self.model is None:
            print("🔄 Loading AraBERT model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(ARABERT_MODEL_PATH).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL_PATH)
            self.model.eval()
            print("✅ AraBERT model loaded!")
    
    def predict(self, text):
        """Predict using transformers cleaning approach."""
        cleaned = clean_arabic_text_transformers(text)
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = softmax(logits.cpu().numpy()[0])
        predicted_class = "Fake" if probs[1] > probs[0] else "Real"
        confidence = float(max(probs))
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {"real": float(probs[0]), "fake": float(probs[1])},
            "cleaned_text": cleaned
        }

def choose_model():
    """Let user choose between ML and AraBERT models."""
    print("\n🇵🇸 Palestine Fake News Detection")
    print("=" * 50)
    print("Choose a model:")
    print("  1️⃣  ML Approach (XGBoost + Minimal Cleaning) - Winner Model")
    print("  2️⃣  AraBERT Approach (Transformer)")
    print("=" * 50)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            return 'ml'
        elif choice == '2':
            return 'arabert'
        else:
            print("⚠️  Please enter 1 or 2")

def main():
    model_choice = choose_model()
    
    # Initialize predictors (but don't load models yet)
    ml_predictor = MLPredictor()
    arabert_predictor = AraBERTPredictor()
    
    # Load only the chosen model
    if model_choice == 'ml':
        predictor = ml_predictor
        predictor.load_model()
        model_name = "XGBoost (Minimal Cleaning)"
    else:
        predictor = arabert_predictor
        predictor.load_model()
        model_name = "AraBERT (Transformers Cleaning)"
    
    print(f"\n🎯 Using: {model_name}")
    print("=" * 50)
    print("Commands: 'quit' or 'q' to exit")
    print("=" * 50)
    
    while True:
        text = input(f"\nEnter Arabic news text: ").strip()
        
        if text.lower() in ['quit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not text:
            print("⚠️  Please enter some text.")
            continue
        
        print("\n🔍 Analyzing...")
        print("-" * 30)
        
        try:
            result = predictor.predict(text)
            
            print(f"📊 Result: {result['predicted_class']}")
            print(f"🎯 Confidence: {result['confidence']:.1%}")
            print(f"📈 Probabilities - Real: {result['probabilities']['real']:.1%}, Fake: {result['probabilities']['fake']:.1%}")
            print(f"📝 Cleaned: {result['cleaned_text'][:100]}{'...' if len(result['cleaned_text']) > 100 else ''}")
            
            if result['predicted_class'] == 'Real':
                print("✅ This appears to be REAL news")
            else:
                print("⚠️  This appears to be FAKE news")
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
