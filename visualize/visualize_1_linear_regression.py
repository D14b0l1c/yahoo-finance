"""
Visualize predictions from the Linear Regression model saved in /models.
This script loads the model and prints coefficients and example predictions.
"""

import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "linear_regression.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path).dropna(subset=["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"])

# Prepare feature matrix
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y_true = df["Current Price"]
y_pred = model.predict(X)

# Show model details
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Show comparison of actual vs predicted prices
comparison = df[["Ticker", "Current Price"]].copy()
comparison["Predicted Price"] = y_pred
print("\nActual vs Predicted Prices (sample):")
print(comparison.head(10))
