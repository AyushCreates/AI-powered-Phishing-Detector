import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Load model and scaler
MODEL_PATH = "models/phishing_model.joblib"
SCALER_PATH = "models/scaler.joblib"
LOG_PATH = "data/predictions.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# App title
st.title("🛡️ AI-Powered Phishing Detector")

# Sidebar navigation
menu = ["🔍 Predict", "📊 View Logs", "ℹ️ About"]
choice = st.sidebar.radio("Navigation", menu)

# Prediction Page
if choice == "🔍 Predict":
    st.subheader("Make Predictions")

    option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

    # --- Manual Entry ---
    if option == "Manual Entry":
        features = st.text_area("Enter comma-separated feature values (48 numbers):")

        if st.button("Predict (Manual)"):
            try:
                values = [float(x) for x in features.split(",")]
                if len(values) != 48:
                    st.error(f"❌ Expected 48 features, but got {len(values)}")
                else:
                    X = np.array(values).reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    prediction = model.predict(X_scaled)[0]

                    result = "Phishing 🚨" if prediction == 1 else "Legit ✅"
                    st.success(f"Prediction: {result}")

                    # Save log
                    os.makedirs("data", exist_ok=True)
                    log_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "features": features,
                        "prediction": result,
                    }
                    if os.path.exists(LOG_PATH):
                        logs = pd.read_csv(LOG_PATH)
                        logs = pd.concat([logs, pd.DataFrame([log_entry])], ignore_index=True)
                    else:
                        logs = pd.DataFrame([log_entry])
                    logs.to_csv(LOG_PATH, index=False)

            except Exception as e:
                st.error(f"Error: {e}")

    # --- CSV Upload ---
    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with 48 feature columns", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if df.shape[1] != 48:
                    st.error(f"❌ Expected 48 columns, but got {df.shape[1]}")
                else:
                    X_scaled = scaler.transform(df.values)
                    preds = model.predict(X_scaled)

                    df["Prediction"] = ["Phishing 🚨" if p == 1 else "Legit ✅" for p in preds]
                    st.dataframe(df)

                    # Save log for all rows
                    os.makedirs("data", exist_ok=True)
                    logs = []
                    for i, row in df.iterrows():
                        logs.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "features": row.drop("Prediction").to_list(),
                            "prediction": row["Prediction"],
                        })

                    if os.path.exists(LOG_PATH):
                        existing = pd.read_csv(LOG_PATH)
                        logs_df = pd.concat([existing, pd.DataFrame(logs)], ignore_index=True)
                    else:
                        logs_df = pd.DataFrame(logs)

                    logs_df.to_csv(LOG_PATH, index=False)

                    st.success("✅ Predictions completed & logs updated")
                    st.download_button("⬇️ Download Results", df.to_csv(index=False), "predictions_with_results.csv")

            except Exception as e:
                st.error(f"Error: {e}")

# Logs Page
elif choice == "📊 View Logs":
    st.subheader("Prediction Logs")
    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH)
        st.dataframe(logs)
        st.download_button("⬇️ Download Logs", data=logs.to_csv(index=False), file_name="predictions.csv")
    else:
        st.warning("No logs available yet. Run a prediction first!")

# About Page
elif choice == "ℹ️ About":
    st.subheader("About This Project")
    st.write("""
    This is an **AI-Powered Phishing Detector** built with:
    - Python 🐍
    - Scikit-learn 🤖
    - Streamlit 🌐

    You can either:
    - Enter **48 feature values manually** (comma-separated)
    - Or **upload a CSV file** with multiple rows for batch predictions

    Results are stored in logs for analysis 📊
    """)
