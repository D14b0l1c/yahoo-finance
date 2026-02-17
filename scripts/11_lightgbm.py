"""
LightGBM Regressor
Use for: Stock Price Prediction using fundamentals
Input: Financial metrics (PE, Market Cap, etc.)
Strength: Fast training, handles large datasets, often outperforms XGBoost
Limitation: Can overfit on small datasets
"""

import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")
model_path = os.path.join(BASE_DIR, "models", "lightgbm_regressor.pkl")

# Load data
df = pd.read_csv(data_path)

# Define features (adjust based on your data columns)
feature_cols = [col for col in df.columns if col not in ['Ticker', 'Current Price', 'Company Name']]
feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found in dataset")

X = df[feature_cols].fillna(0)
y = df["Current Price"].fillna(df["Current Price"].median())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"LightGBM R² - Train: {train_score:.4f}, Test: {test_score:.4f}")

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"LightGBM model saved to: {model_path}")
