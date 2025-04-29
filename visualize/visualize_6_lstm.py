"""
Visualize LSTM forecast vs. recent closing prices.
"""

import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "lstm_forecast.keras")

# Load model
model = load_model(model_path)

# Load AAPL data
ticker = "AAPL"
df = yf.download(ticker, period="6mo", interval="1d")[["Close"]].dropna()

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Prepare sequence
input_seq = scaled[-10:]
input_seq = input_seq.reshape((1, 10, 1))

# Predict next price
pred_scaled = model.predict(input_seq, verbose=0)
predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

# Append prediction to recent closing prices for plotting
plot_df = df.tail(10).copy()
plot_df["Index"] = range(len(plot_df))
future_index = plot_df["Index"].max() + 1
plot_df = plot_df.set_index("Index")
plot_df.loc[future_index] = [predicted_price]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(plot_df.index[:-1], plot_df["Close"].iloc[:-1], marker='o', label="Actual Close")
plt.plot(future_index, predicted_price, marker='x', color='red', label="LSTM Prediction")
plt.title(f"{ticker} - LSTM Next-Day Forecast")
plt.xlabel("Index")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()