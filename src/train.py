import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load processed dataset
X_train, X_test, y_train, y_test = joblib.load("data/processed/split_data.joblib")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Initialize model with class balancing for imbalanced dataset
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # probability for positive class

# Evaluation
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC for imbalanced datasets
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")
except:
    print("\nROC-AUC Score could not be calculated (check test set size).")

# Save trained model
model_path = "models/phishing_model.joblib"
joblib.dump(clf, model_path)
print(f"\nModel training complete. Saved as {model_path}")
