"""
Visualize predictions from the XGBoost model saved in /models.
This script loads the model and compares predictions to actual prices.
"""

import pandas as pd
import joblib
import os

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "xgboost_regressor.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path).dropna(subset=["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"])

# Run predictions
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
df["Predicted Price"] = model.predict(X)

# Show comparison
print("XGBoost Predictions vs Actual:")
print(df[["Ticker", "Current Price", "Predicted Price"]].head(10))
