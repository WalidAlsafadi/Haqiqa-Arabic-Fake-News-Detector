import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# Model configuration
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
model_path = "WalidAlsafadi/arabert-fake-news-detector"  # Your Hugging Face model

class ArabicFakeNewsDetector:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned AraBERT model"""
        try:
            # Load tokenizer and model from Hugging Face Hub
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to base model for demo purposes
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, text):
        """Predict if Arabic news text is real or fake"""
        if not text or text.strip() == "":
            return {"error": "النص فارغ. يرجى إدخال نص للتحليل."}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to percentages
            fake_prob = float(probabilities[0][0]) * 100
            real_prob = float(probabilities[0][1]) * 100
            
            # Determine prediction
            prediction = "حقيقي" if real_prob > fake_prob else "مزيف"
            confidence = max(real_prob, fake_prob)
            
            return {
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "Real": round(real_prob, 2),
                "Fake": round(fake_prob, 2)
            }
            
        except Exception as e:
            return {"error": f"خطأ في التحليل: {str(e)}"}

# Initialize detector
detector = ArabicFakeNewsDetector()

def predict_news(text):
    """Gradio interface function"""
    result = detector.predict(text)
    
    if "error" in result:
        return result["error"]
    
    # Format output for Gradio
    output = f"""
    **التنبؤ:** {result['prediction']}
    **الثقة:** {result['confidence']:.1f}%
    
    **التفاصيل:**
    - احتمالية الخبر الحقيقي: {result['Real']:.1f}%
    - احتمالية الخبر المزيف: {result['Fake']:.1f}%
    """
    
    return output

# Create Gradio interface
with gr.Blocks(
    title="كاشف الأخبار المزيفة العربية - AraBERT",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        direction: rtl;
        text-align: right;
    }
    .gr-textbox textarea {
        direction: rtl;
        text-align: right;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🔍 كاشف الأخبار المزيفة العربية
    ### مدعوم بنموذج AraBERT المُدرب على الأخبار الفلسطينية
    
    أدخل نص الخبر العربي للحصول على تنبؤ حول صحته
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="نص الخبر العربي",
                placeholder="أدخل نص الخبر هنا...",
                lines=5,
                rtl=True
            )
            
            predict_btn = gr.Button(
                "تحليل الخبر",
                variant="primary"
            )
    
    with gr.Row():
        output = gr.Markdown(
            label="النتيجة",
            rtl=True
        )
    
    # Event handlers
    predict_btn.click(
        fn=predict_news,
        inputs=[text_input],
        outputs=[output]
    )
    
    text_input.submit(
        fn=predict_news,
        inputs=[text_input],
        outputs=[output]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["أعلنت وزارة الصحة الفلسطينية عن تسجيل حالات إصابة جديدة بفيروس كورونا في قطاع غزة"],
            ["شركة تقنية جديدة تعلن عن اختراق علمي مذهل سيغير العالم خلال أسبوع"],
            ["الجامعة الإسلامية في غزة تستقبل وفداً أكاديمياً من جامعة الأزهر"]
        ],
        inputs=[text_input],
        outputs=[output],
        fn=predict_news,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )