"""
Explainable Boosting Machine (EBM)
Use for: Transparent regression modeling of stock prices
Input: PE Ratio, ROE, Beta, EPS
Output: Predicted current price
"""

import pandas as pd
import os
import joblib
from interpret.glassbox import ExplainableBoostingRegressor

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "ebm.pkl")

# Load and prepare data
df = pd.read_csv(data_path)
df = df.dropna(subset=["PE Ratio", "Return on Equity", "Beta", "EPS", "Current Price"])
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y = df["Current Price"]

# Train EBM
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(ebm, model_path)
print(f"EBM model trained and saved to: {model_path}")