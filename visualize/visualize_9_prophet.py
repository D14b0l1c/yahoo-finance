"""
Visualize Prophet forecast results.
This script loads the Prophet model and forecasts future prices.
"""

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "prophet_forecast.pkl")

# Load model
model = joblib.load(model_path)

# Create future dataframe for prediction (30 days ahead)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Display forecast
print("Prophet 30-day Forecast:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# Plot forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                alpha=0.3, color='blue', label='Confidence Interval')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Prophet Price Forecast')
ax.legend()
plt.tight_layout()
plt.show()
