"""
Random Forest
Use for: Predicting prices or classifying sentiment
Input: Fundamentals like PE, ROE, Beta
Strength: Non-linear, handles noisy data
Limitation: Slower, harder to scale
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "random_forest_regressor.pkl")

# Load data
df = pd.read_csv(data_path)
required = ["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"]
df = df.dropna(subset=required)

# Features and target
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y = df["Current Price"]

# Train and save model
model = RandomForestRegressor().fit(X, y)
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Random Forest Regressor model saved to: {model_path}")