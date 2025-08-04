"""
AraBERT inference module for Palestinian fake news detection.
Provides easy-to-use interface for making predictions.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from typing import Union, List, Dict


class ArabertPredictor:
    """
    AraBERT predictor for fake news detection.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained AraBERT model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        print(f"AraBERT model loaded successfully on {self.device}")
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict whether news text is real or fake.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Dictionary with predictions or list of dictionaries
        """
        if isinstance(text, str):
            return self._predict_single(text)
        elif isinstance(text, list):
            return [self._predict_single(t) for t in text]
        else:
            raise ValueError("Input must be string or list of strings")
    
    def _predict_single(self, text: str) -> Dict:
        """
        Predict for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Convert to probabilities
        probs = softmax(logits.cpu().numpy()[0])
        
        # Determine prediction
        predicted_label = 1 if probs[1] > probs[0] else 0
        predicted_class = "Fake" if predicted_label == 1 else "Real"
        confidence = float(max(probs))
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "real": float(probs[0]),
                "fake": float(probs[1])
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict for a batch of texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Convert to probabilities
            probs = softmax(logits.cpu().numpy(), axis=1)
            
            # Process each prediction in batch
            for j, text in enumerate(batch_texts):
                predicted_label = 1 if probs[j][1] > probs[j][0] else 0
                predicted_class = "Fake" if predicted_label == 1 else "Real"
                confidence = float(max(probs[j]))
                
                results.append({
                    "text": text,
                    "predicted_label": predicted_label,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": {
                        "real": float(probs[j][0]),
                        "fake": float(probs[j][1])
                    }
                })
        
        return results


def load_arabert_predictor(model_path: str) -> ArabertPredictor:
    """
    Load AraBERT predictor.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Initialized predictor instance
    """
    return ArabertPredictor(model_path)


if __name__ == "__main__":
    # Interactive AraBERT inference
    print("=" * 60)
    print("Palestine Fake News Detection - AraBERT Inference")
    print("=" * 60)
    
    try:
        predictor = ArabertPredictor("models/arabert/finetuned-model")
        
        while True:
            print("\nEnter Arabic news text to analyze (or 'quit' to exit):")
            user_text = input("> ")
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_text.strip():
                print("Please enter some text.")
                continue
            
            # Make prediction
            result = predictor.predict(user_text)
            
            # Display results
            print("\n" + "=" * 50)
            print("DETECTION RESULT:")
            print("=" * 50)
            print(f"🔍 Classification: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"📈 Real News Probability: {result['probabilities']['real']:.1%}")
            print(f"📈 Fake News Probability: {result['probabilities']['fake']:.1%}")
            
            # Add emoji indicator
            if result['predicted_class'] == 'Real':
                print("✅ This appears to be REAL news")
            else:
                print("⚠️  This appears to be FAKE news")
                
    except FileNotFoundError:
        print("❌ Error: AraBERT model not found at 'models/arabert/finetuned-model'")
        print("Please train the model first using: python run_arabert_training.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
