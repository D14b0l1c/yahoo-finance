"""
Visualize EBM predictions vs actuals for stock price regression.
"""

import pandas as pd
import joblib
import os

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "ebm.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
ebm = joblib.load(model_path)
df = pd.read_csv(data_path)
df = df.dropna(subset=["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"])

# Prepare features
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
df["Predicted Price"] = ebm.predict(X)

# Compare predictions
print("EBM Predictions vs Actuals:")
print(df[["Ticker", "Current Price", "Predicted Price"]].head(10))
