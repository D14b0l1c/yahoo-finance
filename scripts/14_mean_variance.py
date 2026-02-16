"""
Mean-Variance Optimization (Markowitz Portfolio Theory)
Use for: Portfolio Allocation and Optimization
Input: Historical returns for multiple assets
Strength: Optimal risk/return tradeoff, foundational in finance
Limitation: Sensitive to input estimates, assumes normal distributions
"""

import pandas as pd
import numpy as np
import joblib
import os
from scipy.optimize import minimize

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")
model_path = os.path.join(BASE_DIR, "models", "portfolio_optimization.pkl")

# Load data
df = pd.read_csv(data_path)

# For portfolio optimization, we need returns for multiple assets
# Using available data to demonstrate the concept
if "Current Price" not in df.columns:
    raise ValueError("Current Price column missing from tech_stocks_data.csv")

# Simulate multi-asset scenario using available data
# In production, you'd load actual historical returns per ticker
prices = df["Current Price"].dropna().values
n_assets = min(10, len(prices))  # Use up to 10 "assets"

# Generate synthetic covariance matrix (in production, use actual returns)
np.random.seed(42)
returns = np.random.randn(252, n_assets) * 0.02 + 0.0005  # Daily returns

# Calculate expected returns and covariance
expected_returns = np.mean(returns, axis=0) * 252  # Annualized
cov_matrix = np.cov(returns.T) * 252  # Annualized covariance

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def portfolio_return(weights, expected_returns):
    return weights.T @ expected_returns

def negative_sharpe(weights, expected_returns, cov_matrix, risk_free_rate=0.04):
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return -(ret - risk_free_rate) / vol

# Optimize for maximum Sharpe ratio
n = len(expected_returns)
initial_weights = np.ones(n) / n
bounds = tuple((0, 1) for _ in range(n))
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(
    negative_sharpe,
    initial_weights,
    args=(expected_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights, expected_returns)
optimal_vol = portfolio_volatility(optimal_weights, cov_matrix)
sharpe_ratio = (optimal_return - 0.04) / optimal_vol

# Save optimization results
optimization_results = {
    'optimal_weights': optimal_weights,
    'expected_return': optimal_return,
    'volatility': optimal_vol,
    'sharpe_ratio': sharpe_ratio,
    'expected_returns': expected_returns,
    'cov_matrix': cov_matrix
}

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(optimization_results, model_path)

print("Portfolio Optimization Results:")
print(f"  Expected Annual Return: {optimal_return*100:.2f}%")
print(f"  Annual Volatility: {optimal_vol*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"\nOptimal Weights: {optimal_weights.round(3)}")
print(f"\nResults saved to: {model_path}")
