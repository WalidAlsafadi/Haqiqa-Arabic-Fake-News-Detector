import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from src.text_cleaner import clean_arabic_text

# Load model and vectorizer
MODEL_PATH = "models/XGBoost.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

print("🔹 Loading model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("✅ Ready. Type a news text to classify.")
print("=" * 50)

while True:
    try:
        user_input = input("\n📝 Enter text (or 'exit' to quit):\n> ").strip()
        if user_input.lower() == "exit":
            break

        # Clean + vectorize
        clean_text = clean_arabic_text(user_input)
        text_vector = vectorizer.transform([clean_text])

        # Predict
        pred = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0][1]

        label = "FAKE" if pred == 1 else "REAL"
        print(f"✅ Prediction: {label} (Fake Prob: {prob:.2%})")

    except Exception as e:
        print("⚠️ Error:", e)
