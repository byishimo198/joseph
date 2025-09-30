# ================================
# Streamlit Deployment: Smart Cold Room Monitoring System
# ================================

import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("coldroom_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title
st.title("❄️ Smart Cold Room Monitoring System")

st.markdown("""
This app predicts whether the cold room is in a **Normal** or **Abnormal Condition**  
based on **Temperature (°C)** and **Humidity (%)**.
""")

# User Inputs
temp = st.number_input("🌡️ Enter Temperature (°C)", min_value=-10.0, max_value=20.0, step=0.1)
hum = st.number_input("💧 Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

# Prediction
if st.button("🔍 Predict Condition"):
    features = scaler.transform([[temp, hum]])
    prediction = model.predict(features)[0]

    if prediction == 0:
        st.success("✅ Normal Condition")
    else:
        st.error("⚠ Abnormal Condition Detected")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Built with Streamlit, Scikit-Learn & Random Forest Classifier")
