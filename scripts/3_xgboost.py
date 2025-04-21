"""
XGBoost
Use for: Forecasting price or analyst rating
Input: Any tabular data (fundamentals, tech indicators)
Strength: High accuracy, robust
Limitation: Harder to explain without SHAP
"""

import pandas as pd
from xgboost import XGBRegressor
import joblib
import os

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "xgboost_regressor.pkl")

# Load and validate data
df = pd.read_csv(data_path)
required = ["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"]
df = df.dropna(subset=required)

# Train model
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y = df["Current Price"]
model = XGBRegressor(objective="reg:squarederror").fit(X, y)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"XGBoost Regressor model saved to: {model_path}")