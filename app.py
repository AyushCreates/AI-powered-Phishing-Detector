import streamlit as st
import numpy as np
import joblib
import base64
import io
from feature_extractor import extract_features

st.set_page_config(page_title="AI Phishing Detector", layout="centered")
st.title("🚨 AI-powered Phishing Detector")
st.write("Enter a URL below to check if it’s safe or phishing.")

# ----------------------
# Load Model & Scaler from Secrets
# ----------------------
@st.cache_resource
def load_model_and_scaler():
    # Load model
    model_b64 = st.secrets["models"]["phishing_model"]
    model_bytes = base64.b64decode(model_b64)
    model = joblib.load(io.BytesIO(model_bytes))

    # Load scaler
    scaler_b64 = st.secrets["models"]["scaler"]
    scaler_bytes = base64.b64decode(scaler_b64)
    scaler = joblib.load(io.BytesIO(scaler_bytes))

    return model, scaler

model, scaler = load_model_and_scaler()

# ----------------------
# App UI
# ----------------------
url_input = st.text_input("Enter URL here:")

if st.button("Check URL"):
    if not url_input:
        st.warning("Please enter a URL!")
    else:
        # Extract features automatically from URL
        features = extract_features(url_input)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            st.error("⚠️ This URL is phishing!")
        else:
            st.success("✅ This URL looks safe!")

st.write("---")
st.write("Powered by Streamlit | Model and scaler loaded securely from secrets")
