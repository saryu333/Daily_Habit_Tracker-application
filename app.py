import streamlit as st
import joblib
import os

st.set_page_config(page_title="Daily Habits Mood Predictor")

st.title("🧠 Daily Habits Mood Predictor")

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(BASE_DIR, "lr_model_final.pkl"))
    rf = joblib.load(os.path.join(BASE_DIR, "rf_model_final.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "label_encoder_final.pkl"))
    return lr, rf, le

lr_model, rf_model, label_encoder = load_models()
