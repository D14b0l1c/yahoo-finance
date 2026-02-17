"""
Visualize LightGBM regression results.
This script loads the LightGBM model and displays predictions vs actual.
"""

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "lightgbm_regressor.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Prepare features (same as training)
feature_cols = [col for col in df.columns if col not in ['Ticker', 'Current Price', 'Company Name']]
feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]
X = df[feature_cols].fillna(0)
y_actual = df["Current Price"].fillna(df["Current Price"].median())

# Predict
y_pred = model.predict(X)

# Display feature importance
print("LightGBM Feature Importance:")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.head(10))

# Plot predictions vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(y_actual, y_pred, alpha=0.6)
axes[0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label='Perfect Prediction')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title('LightGBM: Predicted vs Actual')
axes[0].legend()

# Feature importance
top_features = importance.head(10)
axes[1].barh(top_features['Feature'], top_features['Importance'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
