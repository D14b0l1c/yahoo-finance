"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
Use for: Volatility Modeling and Risk Analysis
Input: Historical returns (price changes)
Strength: Models time-varying volatility, essential for risk management
Limitation: Assumes specific distributional properties
"""

import pandas as pd
import numpy as np
import joblib
import os
from arch import arch_model

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")
model_path = os.path.join(BASE_DIR, "models", "garch_volatility.pkl")

# Load data
df = pd.read_csv(data_path)

# Validate required columns
if "Current Price" not in df.columns:
    raise ValueError("Current Price column missing from portfolio_data.csv")

# Calculate returns (percentage change)
prices = df["Current Price"].dropna()
returns = prices.pct_change().dropna() * 100  # Scale for numerical stability

# Fit GARCH(1,1) model - standard specification
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
model_fit = model.fit(disp='off')

# Save fitted model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model_fit, model_path)
print(f"GARCH volatility model saved to: {model_path}")
print("\nModel Summary:")
print(model_fit.summary())
