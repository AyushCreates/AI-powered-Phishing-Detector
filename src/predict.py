import os
import sys
import joblib
import pandas as pd

# Paths
SCALER_PATH = "models/scaler.joblib"
MODEL_PATH = "models/phishing_model.joblib"

# Load scaler and model
if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
    print("Model or scaler not found. Please run preprocess.py and train.py first.")
    sys.exit(1)

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

def predict_from_csv(csv_path):
    """Predict phishing/legit from a CSV file with feature columns"""
    df = pd.read_csv(csv_path)
    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)
    df["Prediction"] = ["Phishing" if p == 1 else "Legit" for p in predictions]
    print(df.head())
    output_path = "data/predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def predict_from_features(features):
    """Predict from a single feature vector (list of numbers)"""
    X_scaled = scaler.transform([features])
    prediction = model.predict(X_scaled)[0]
    return "Phishing" if prediction == 1 else "Legit"

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".csv"):
        # Example: python src/predict.py data/sample_urls.csv
        predict_from_csv(sys.argv[1])
    else:
        # Example: python src/predict.py
        print("Interactive mode: Enter features manually")
        print("For now, enter the exact number of features (same as training set).")

        try:
            raw = input("Enter comma-separated feature values:\n")
            features = [float(x) for x in raw.split(",")]
            result = predict_from_features(features)
            print("Prediction:", result)
        except Exception as e:
            print("❌ Error:", e)
