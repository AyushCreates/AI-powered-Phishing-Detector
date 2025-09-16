import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# Drop unnecessary columns
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Separate features and target
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save processed data and scaler
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "data/processed/split_data.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("Data preprocessing complete. Saved split dataset and scaler.")
