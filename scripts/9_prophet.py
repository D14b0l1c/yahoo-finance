"""
Prophet (Facebook/Meta)
Use for: Time Series Forecasting with seasonality
Input: Historical closing prices with dates
Strength: Handles seasonality, holidays, and trends automatically
Limitation: Requires date column, can be slow on large datasets
"""

import pandas as pd
import joblib
import os
from prophet import Prophet

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")
model_path = os.path.join(BASE_DIR, "models", "prophet_forecast.pkl")

# Load data
df = pd.read_csv(data_path)

# Validate required columns
if "Current Price" not in df.columns:
    raise ValueError("Current Price column missing from portfolio_data.csv")

# Prophet requires 'ds' (date) and 'y' (value) columns
# Create synthetic dates if no date column exists
price_series = df["Current Price"].dropna().reset_index(drop=True)
prophet_df = pd.DataFrame({
    'ds': pd.date_range(start='2024-01-01', periods=len(price_series), freq='D'),
    'y': price_series.values
})

# Train Prophet model
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(prophet_df)

# Save fitted model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Prophet forecast model saved to: {model_path}")
