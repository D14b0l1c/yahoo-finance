"""
Visualize ARIMA forecast results.
This script loads the ARIMA model and forecasts the next 5 days.
"""

import pandas as pd
import joblib
import os

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "arima_forecast.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model
model_fit = joblib.load(model_path)

# Make forecast
forecast = model_fit.forecast(steps=5)

# Display forecast
print("5-day ARIMA forecast:")
print(forecast)
