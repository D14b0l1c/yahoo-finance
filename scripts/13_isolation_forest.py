"""
Isolation Forest
Use for: Anomaly Detection in stock prices/returns
Input: Price data or financial metrics
Strength: Detects unusual market conditions, outliers, potential manipulation
Limitation: Unsupervised - requires interpretation of results
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")
model_path = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")

# Load data
df = pd.read_csv(data_path)

# Select numeric columns for anomaly detection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Current Price' in numeric_cols:
    # Use price and any other numeric features
    X = df[numeric_cols].fillna(0)
else:
    raise ValueError("Current Price column missing from portfolio_data.csv")

# Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expect ~10% anomalies
    random_state=42,
    n_jobs=-1
)
model.fit(X)

# Predict anomalies (-1 = anomaly, 1 = normal)
predictions = model.predict(X)
anomaly_count = (predictions == -1).sum()

print(f"Isolation Forest trained on {len(X)} samples")
print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Isolation Forest model saved to: {model_path}")
