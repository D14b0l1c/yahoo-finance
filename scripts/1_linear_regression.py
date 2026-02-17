"""
Linear Regression
Use for: Predicting next-day price (basic)
Input: EPS, PE Ratio, Days
Strength: Simple, transparent
Limitation: Too rigid for market volatility

Useful for learning, but too basic to capture real market behavior.
It’s the most mechanical form of TA — and often arbitraged away by the market.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Dynamically resolve path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")
model_path = os.path.join(BASE_DIR, "models", "linear_regression.pkl")

# Load dataset
df = pd.read_csv(data_path)
print("Available columns:", df.columns.tolist())  # Debug line

# Ensure required columns exist
required = ["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise KeyError(f"Missing required column(s): {missing}")

# Filter clean rows
df = df.dropna(subset=required)

# Train model
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y = df["Current Price"]

model = LinearRegression().fit(X, y)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print(f"Linear Regression model trained and saved to: {model_path}")
