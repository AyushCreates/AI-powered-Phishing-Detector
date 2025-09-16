import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ========================
# Paths
# ========================
MODEL_PATH = "models/phishing_model.joblib"
SCALER_PATH = "models/scaler.joblib"
LOG_PATH = "data/predictions.csv"

# ========================
# Load model & scaler
# ========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Ensure log file exists
if not os.path.exists(LOG_PATH):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    pd.DataFrame(columns=["timestamp", "features", "prediction"]).to_csv(LOG_PATH, index=False)

# ========================
# Helper Functions
# ========================
def log_prediction(features, prediction):
    df = pd.read_csv(LOG_PATH)
    new_row = {"timestamp": datetime.now(), "features": features, "prediction": prediction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

def make_prediction(features):
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    pred = model.predict(features_scaled)[0]
    return "Phishing" if pred == 1 else "Legit"

# ========================
# Streamlit App
# ========================
st.set_page_config(page_title="AI-Powered Phishing Detector", layout="wide")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction", "Logs & Dashboard"])

st.title("🛡️ AI-Powered Phishing Detector")

# ========================
# HOME PAGE
# ========================
if page == "Home":
    st.markdown("""
    ### Welcome to the AI-Powered Phishing Detector  
    - Enter feature values manually for single predictions  
    - Upload CSV files for **batch predictions**  
    - View **logs and dashboards** to track results  
    """)

# ========================
# SINGLE PREDICTION
# ========================
elif page == "Single Prediction":
    st.subheader("🔎 Single Prediction")

    feature_input = st.text_input("Enter comma-separated feature values", "")

    if st.button("Predict"):
        try:
            features = list(map(float, feature_input.split(",")))
            prediction = make_prediction(features)
            log_prediction(features, prediction)

            if prediction == "Phishing":
                st.error("⚠️ Prediction: Phishing")
            else:
                st.success("✅ Prediction: Legit")

        except Exception as e:
            st.error(f"Error: {e}")

# ========================
# BATCH PREDICTION
# ========================
elif page == "Batch Prediction":
    st.subheader("📂 Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            features_scaled = scaler.transform(df.values)
            preds = model.predict(features_scaled)
            results = ["Phishing" if p == 1 else "Legit" for p in preds]

            df["Prediction"] = results
            st.dataframe(df)

            # Log results
            for i in range(len(df)):
                log_prediction(df.iloc[i, :-1].tolist(), df.iloc[i, -1])

            # Downloadable CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")

# ========================
# LOGS & DASHBOARD
# ========================
elif page == "Logs & Dashboard":
    st.subheader("📊 Prediction Logs & Dashboard")

    try:
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df.tail(20))  # Show recent 20 logs

        col1, col2 = st.columns(2)

        with col1:
            counts = df["prediction"].value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", color=["green", "red"], ax=ax)
            ax.set_title("Prediction Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            timeline = df.groupby(df["timestamp"].dt.date)["prediction"].count()
            fig2, ax2 = plt.subplots()
            timeline.plot(kind="line", marker="o", ax=ax2)
            ax2.set_title("Predictions Over Time")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error loading logs: {e}")
