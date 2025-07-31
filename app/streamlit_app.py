import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing.text_cleaner import clean_arabic_text_minimal

st.set_page_config(page_title="كاشف الأخبار الكاذبة لفلسطين", layout="centered")

# -----------------------------
# Model & App Config
# -----------------------------
MODEL_PATH = "models/trained/XGBoost_minimal_best_model.pkl"
VECTORIZER_PATH = "models/trained/fitted_vectorizer.pkl"
THRESHOLD = 0.5
MIN_CHARS = 30

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        color: #1abc9c;
        font-weight: bold;
        text-align: center;
    }
    .subtext {
        font-size: 18px;
        color: #ffffff;
        text-align: center;
    }
    .footer {
        font-size: 14px;
        color: #7f8c8d;
        text-align: center;
        padding: 20px 0;
    }
    .footer a {
        color: #3498db;
        text-decoration: none;
        margin: 0 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and fitted vectorizer"""
    try:
        import pickle
        
        # Load the trained model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load the fitted vectorizer
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

def predict_news(text, model, vectorizer):
    """Predict if news is fake or real"""
    try:
        # Clean the text
        cleaned_text = clean_arabic_text_minimal(text)
        
        # Vectorize the text using the trained vectorizer
        X_vectorized = vectorizer.transform([cleaned_text])
        
        # Get prediction probabilities
        prob_fake = model.predict_proba(X_vectorized)[0][1]
        prob_real = 1 - prob_fake
        
        return prob_fake, prob_real, cleaned_text
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# -----------------------------
# Page Header
# -----------------------------
st.markdown("<div class='main-title'>📰 كاشف الأخبار الكاذبة لفلسطين</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>أدخل عنوان خبر أو مقطعًا إخباريًا باللغة العربية لتحديد ما إذا كان <strong>حقيقيًا</strong> أو <strong>كاذبًا</strong>.</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>📌 هذا النموذج قد يُخطئ أحيانًا. يعمل بشكل أفضل مع الأخبار الفلسطينية.</div>", unsafe_allow_html=True)

# -----------------------------
# Session History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# User Input
# -----------------------------
text_input = st.text_area("📝 أدخل الخبر هنا", height=150, key="input")
char_count = len(text_input.strip())
st.caption(f"🔠 عدد الأحرف: {char_count}")

if st.button("تحليل الخبر"):
    if not text_input.strip():
        st.warning("يرجى إدخال نص إخباري باللغة العربية.")
    elif char_count < MIN_CHARS:
        st.warning(f"🔍 للحصول على نتيجة أدق، أدخل {MIN_CHARS} حرفًا على الأقل.")
    elif model is None or vectorizer is None:
        st.error("خطأ في تحميل النموذج. يرجى المحاولة مرة أخرى.")
    else:
        # Use the new prediction pipeline
        prob_fake, prob_real, cleaned = predict_news(text_input, model, vectorizer)
        
        if prob_fake is not None:
            is_fake = int(prob_fake > THRESHOLD)
            label = "❌ كاذب" if is_fake else "✅ حقيقي"
            confidence = f"{prob_real * 100:.2f}% صحيح"

            st.markdown(f"### التقييم: **{label}**")
            st.markdown(f"#### النتيجة: `{confidence}`")

            st.session_state.history.append({
                "النص المُدخل": text_input.strip(),
                "التقييم": label,
                "الثقة": confidence
            })

            with st.expander("🔍 النص بعد التنظيف"):
                st.write(cleaned)

# -----------------------------
# Prediction History
# -----------------------------
if st.session_state.history:
    st.subheader("🧾 السجل (آخر 10 نتائج)")
    df_hist = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(df_hist[::-1], use_container_width=True)
