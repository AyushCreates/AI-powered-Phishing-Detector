import os
import pandas as pd

# Get absolute path of project root (where this script is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "phishing_dataset_ml")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Find CSV file
csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
phishing_csv = os.path.join(RAW_DIR, csv_files[0])

print("[OK] Using CSV file:", phishing_csv)

# Load dataset
df = pd.read_csv(phishing_csv)
print("[OK] Loaded dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Save processed dataset
processed_path = os.path.join(PROCESSED_DIR, "phishing_dataset_clean.csv")
df.to_csv(processed_path, index=False)
print(f"[OK] Saved processed dataset at: {processed_path}")

# Show class distribution
if "CLASS_LABEL" in df.columns:
    print("\nClass distribution:")
    print(df["CLASS_LABEL"].value_counts())
else:
    print("\n[Warning] CLASS_LABEL column not found in dataset.")

# Create processed folder if not exists
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("Cleaned dataset saved to data/processed/cleaned_dataset.csv")
