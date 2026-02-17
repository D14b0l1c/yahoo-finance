"""
Visualize Logistic Regression predictions.
This script loads the Logistic Regression model and predicts Up/Down movement.
"""

import pandas as pd
import joblib
import os

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Create movement label
df["Price Change"] = df["Current Price"].diff().shift(-1)
df["Movement"] = df["Price Change"].apply(lambda x: 1 if x > 0 else 0)

# Drop missing
df = df.dropna(subset=["PE Ratio", "Return on Equity", "Beta", "EPS", "Movement"])

# Predict
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
df["Predicted Movement"] = model.predict(X)

# Show comparison
print("Logistic Regression Predictions (1=Up, 0=Down):")
print(df[["Ticker", "Movement", "Predicted Movement"]].head(10))
