"""
LSTM
Use for: Multi-day sequential price forecasting
Input: Rolling window of past closing prices
Strength: Captures time-dependent patterns and momentum
Limitation: Requires normalization and careful tuning
"""

import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "lstm_forecast.keras")

# Load historical data from Yahoo Finance
ticker = "AAPL"  # example ticker
df = yf.download(ticker, period="6mo", interval="1d")
df = df[["Close"]].dropna()

# Normalize closing prices
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Prepare sequences (10 days history -> 1 day prediction)
X, y = [], []
for i in range(10, len(scaled)):
    X.append(scaled[i-10:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train with early stopping
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2,
          callbacks=[EarlyStopping(patience=5)], verbose=0)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"LSTM model trained and saved to: {model_path}")