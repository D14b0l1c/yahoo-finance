"""
Visualize GARCH volatility model results.
This script loads the GARCH model and forecasts future volatility.
"""

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "garch_volatility.pkl")

# Load model
model_fit = joblib.load(model_path)

# Forecast volatility for next 10 days
forecast = model_fit.forecast(horizon=10)
volatility_forecast = np.sqrt(forecast.variance.values[-1])

print("GARCH 10-day Volatility Forecast:")
for i, vol in enumerate(volatility_forecast, 1):
    print(f"  Day {i}: {vol:.4f}%")

# Plot conditional volatility
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Historical conditional volatility
cond_vol = model_fit.conditional_volatility
axes[0].plot(cond_vol, color='blue')
axes[0].set_title('Historical Conditional Volatility')
axes[0].set_xlabel('Observation')
axes[0].set_ylabel('Volatility (%)')

# Forecasted volatility
axes[1].bar(range(1, 11), volatility_forecast, color='orange')
axes[1].set_title('10-Day Volatility Forecast')
axes[1].set_xlabel('Days Ahead')
axes[1].set_ylabel('Volatility (%)')

plt.tight_layout()
plt.show()
