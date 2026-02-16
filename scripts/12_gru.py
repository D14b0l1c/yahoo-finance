"""
GRU (Gated Recurrent Unit)
Use for: Sequential Stock Price Prediction
Input: Historical price sequences
Strength: Faster than LSTM, similar performance, less prone to overfitting
Limitation: May miss very long-term dependencies
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "gru_forecast.keras")

# Load data
df = pd.read_csv(data_path)

# Validate required columns
if "Current Price" not in df.columns:
    raise ValueError("Current Price column missing from tech_stocks_data.csv")

# Prepare data
prices = df["Current Price"].dropna().values.reshape(-1, 1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Create sequences (lookback of 10 steps)
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = min(10, len(prices_scaled) - 2)
X, y = create_sequences(prices_scaled, seq_length)

if len(X) < 5:
    print("Warning: Not enough data for GRU training. Need more price history.")
else:
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build GRU model
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"GRU forecast model saved to: {model_path}")
