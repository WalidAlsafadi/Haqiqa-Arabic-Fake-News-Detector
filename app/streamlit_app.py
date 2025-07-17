import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd
from src.preprocessing.cleaning import clean_arabic_text_minimal

st.set_page_config(page_title="كاشف الأخبار الكاذبة لفلسطين", layout="centered")

# -----------------------------
# Model & App Config
# -----------------------------
MODEL_PATH = "models/model_final_xgb_pipeline.pkl"
THRESHOLD = 0.45
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
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

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
    else:
        cleaned = clean_arabic_text_minimal(text_input)
        prob_fake = model.predict_proba([cleaned])[0][1]
        prob_real = 1 - prob_fake
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
