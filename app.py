import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Daily Habits Mood Predictor",
    layout="centered"
)

st.title("🧠 Daily Habits Mood Predictor")
st.write("Enter your daily habits to predict your mood")

# Load saved models
lr_model = joblib.load("lr_model_final.pkl")
rf_model = joblib.load("rf_model_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")

st.subheader("📋 Daily Habits Input")

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
study_hours = st.slider("Study Hours", 0.0, 12.0, 4.0)
steps = st.number_input("Steps Walked", min_value=0, max_value=30000, value=6000)
water_intake = st.slider("Water Intake (Liters)", 0.0, 5.0, 2.0)
screen_time = st.slider("Screen Time (Hours)", 0.0, 15.0, 5.0)

input_data = np.array([[sleep_hours, study_hours, steps, water_intake, screen_time]])

if st.button("Predict Mood"):
    lr_pred = lr_model.predict(input_data)
    rf_pred = rf_model.predict(input_data)

    lr_mood = label_encoder.inverse_transform(lr_pred)[0]
    rf_mood = label_encoder.inverse_transform(rf_pred)[0]

    st.success(f"🔍 Logistic Regression Prediction: **{lr_mood}**")
    st.success(f"🌳 Random Forest Prediction: **{rf_mood}**")

st.markdown("---")
st.caption("Daily Habits Tracker – Machine Learning Project")


