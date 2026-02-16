"""
Visualize Mean-Variance Portfolio Optimization results.
This script loads the optimization results and displays the efficient frontier.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "portfolio_optimization.pkl")

# Load optimization results
results = joblib.load(model_path)

optimal_weights = results['optimal_weights']
expected_return = results['expected_return']
volatility = results['volatility']
sharpe_ratio = results['sharpe_ratio']
exp_returns = results['expected_returns']
cov_matrix = results['cov_matrix']

print("Portfolio Optimization Results:")
print(f"  Expected Annual Return: {expected_return*100:.2f}%")
print(f"  Annual Volatility: {volatility*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"\nOptimal Portfolio Weights:")
for i, w in enumerate(optimal_weights):
    if w > 0.01:  # Only show significant weights
        print(f"  Asset {i+1}: {w*100:.2f}%")

# Generate efficient frontier
def portfolio_stats(weights, exp_returns, cov_matrix):
    ret = np.dot(weights, exp_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

n_portfolios = 5000
n_assets = len(exp_returns)
results_array = np.zeros((n_portfolios, 3))

for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    ret, vol = portfolio_stats(weights, exp_returns, cov_matrix)
    results_array[i] = [ret, vol, (ret - 0.04) / vol]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Efficient frontier
scatter = axes[0].scatter(results_array[:, 1]*100, results_array[:, 0]*100, 
                          c=results_array[:, 2], cmap='viridis', alpha=0.5, s=10)
axes[0].scatter(volatility*100, expected_return*100, 
               c='red', marker='*', s=500, label='Optimal Portfolio')
axes[0].set_xlabel('Volatility (%)')
axes[0].set_ylabel('Expected Return (%)')
axes[0].set_title('Efficient Frontier')
axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label='Sharpe Ratio')

# Optimal weights
significant_weights = [(i, w) for i, w in enumerate(optimal_weights) if w > 0.01]
if significant_weights:
    indices, weights = zip(*significant_weights)
    axes[1].bar([f'Asset {i+1}' for i in indices], weights)
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Optimal Portfolio Allocation')
    axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
