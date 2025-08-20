import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os

# Load AraBERT from Hugging Face Hub
ARABERT_MODEL = "WalidAlsafadi/arabert-fake-news-detector"

def load_arabert():
    tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ARABERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_arabert(text, tokenizer, model, device):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    prediction = "Real" if real_prob > fake_prob else "Fake"
    confidence = max(real_prob, fake_prob)
    return {
        "model": "arabert",
        "prediction": prediction,
        "confidence": confidence,
        "real_prob": real_prob,
        "fake_prob": fake_prob
    }

# Load XGBoost model from local file
XGBOOST_PATH = "xgboost_minimal_final.pkl" 

def load_xgboost():
    with open(XGBOOST_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def predict_xgboost(text, model):
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    # Mapping: 0 = Real, 1 = Fake
    prediction = "Real" if pred == 0 else "Fake"
    confidence = max(prob)
    return {
        "model": "xgboost",
        "prediction": prediction,
        "confidence": confidence,
        "real_prob": prob[0],
        "fake_prob": prob[1]
    }

# Load models once
arabert_tokenizer, arabert_model, arabert_device = load_arabert()
xgboost_model = load_xgboost()

def predict(text, model_name):
    if model_name == "arabert":
        return predict_arabert(text, arabert_tokenizer, arabert_model, arabert_device)
    elif model_name == "xgboost":
        return predict_xgboost(text, xgboost_model)
    else:
        return {"error": "Invalid model name. Use 'arabert' or 'xgboost'."}

# Gradio API mode (no UI)
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Dropdown(choices=["arabert", "xgboost"], label="Model")
    ],
    outputs="json",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)