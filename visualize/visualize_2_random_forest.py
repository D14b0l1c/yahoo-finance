"""
Visualize predictions from the Random Forest model saved in /models.
This script loads the model and compares predictions to actual prices.
"""

import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "random_forest_regressor.pkl")
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path).dropna(subset=["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"])

# Prepare input and predictions
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y_pred = model.predict(X)

# Compare with actual prices
comparison = df[["Ticker", "Current Price"]].copy()
comparison["Predicted Price"] = y_pred

print("Sample prediction results:")
print(comparison.head(10))