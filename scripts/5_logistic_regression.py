"""
Logistic Regression
Use for: Predicting next-day price movement (Up/Down)
Input: PE Ratio, Return on Equity, Beta, EPS
Strength: Transparent, fast
Limitation: Limited to simple boundaries
"""

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")

# Load data
df = pd.read_csv(data_path)

# Create label: Up (1) if next day's price is higher, else Down (0)
df["Price Change"] = df["Current Price"].diff().shift(-1)
df["Movement"] = df["Price Change"].apply(lambda x: 1 if x > 0 else 0)

# Validate required fields
required = ["PE Ratio", "Return on Equity", "Beta", "EPS", "Movement"]
df = df.dropna(subset=required)

# Train model
X = df[["PE Ratio", "Return on Equity", "Beta", "EPS"]]
y = df["Movement"]
model = LogisticRegression().fit(X, y)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Logistic Regression model saved to: {model_path}")