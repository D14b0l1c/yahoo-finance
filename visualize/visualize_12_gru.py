"""
Visualize GRU forecast results.
This script loads the GRU model and displays predictions.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "gru_forecast.keras")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
model = load_model(model_path)
df = pd.read_csv(data_path)

# Prepare data (same as training)
prices = df["Current Price"].dropna().values.reshape(-1, 1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Create sequences
seq_length = 10
X, y_actual = [], []
for i in range(len(prices_scaled) - seq_length):
    X.append(prices_scaled[i:i+seq_length])
    y_actual.append(prices_scaled[i+seq_length])
X = np.array(X)
y_actual = np.array(y_actual)

# Predict
y_pred_scaled = model.predict(X, verbose=0)

# Inverse transform
y_actual_inv = scaler.inverse_transform(y_actual)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

print("GRU Forecast Results:")
print(f"  MSE: {np.mean((y_actual_inv - y_pred_inv)**2):.4f}")
print(f"  MAE: {np.mean(np.abs(y_actual_inv - y_pred_inv)):.4f}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_actual_inv, label='Actual', alpha=0.8)
ax.plot(y_pred_inv, label='GRU Prediction', alpha=0.8)
ax.set_xlabel('Time Step')
ax.set_ylabel('Price')
ax.set_title('GRU: Actual vs Predicted Prices')
ax.legend()
plt.tight_layout()
plt.show()
