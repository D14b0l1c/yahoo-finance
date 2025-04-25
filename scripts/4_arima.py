"""
ARIMA
Use for: Time Series Forecasting of stock closing prices
Input: Historical closing prices
Strength: Captures trend and seasonality
Limitation: Doesn't use external features (fundamentals)
"""

import pandas as pd
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "arima_forecast.pkl")

# Load data
df = pd.read_csv(data_path)

# Validate 'Current Price' field
if "Current Price" not in df.columns:
    raise ValueError("Current Price column missing from tech_stocks_data.csv")

# Forecast only using the Current Price series
price_series = df["Current Price"].dropna()

# Simple ARIMA model (p=5, d=1, q=0) - could be optimized further
model = ARIMA(price_series, order=(5, 1, 0))
model_fit = model.fit()

# Save fitted model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model_fit, model_path)
print(f"ARIMA forecast model saved to: {model_path}")