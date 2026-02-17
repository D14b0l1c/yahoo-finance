"""
Visualize Isolation Forest anomaly detection results.
This script loads the model and highlights detected anomalies.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")

# Load model and data
model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Prepare features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].fillna(0)

# Predict anomalies
predictions = model.predict(X)
anomaly_scores = model.decision_function(X)

# Add results to dataframe
df['Anomaly'] = predictions
df['Anomaly_Score'] = anomaly_scores
anomalies = df[df['Anomaly'] == -1]

print(f"Isolation Forest Results:")
print(f"  Total samples: {len(df)}")
print(f"  Anomalies detected: {len(anomalies)}")
print(f"\nAnomalous stocks/entries:")
if 'Ticker' in df.columns:
    print(anomalies[['Ticker', 'Current Price', 'Anomaly_Score']].to_string())
else:
    print(anomalies[['Current Price', 'Anomaly_Score']].head(10))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Anomaly scores distribution
axes[0].hist(anomaly_scores, bins=30, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', label='Threshold')
axes[0].set_xlabel('Anomaly Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Anomaly Score Distribution')
axes[0].legend()

# Scatter plot with anomalies highlighted
if 'Current Price' in df.columns:
    normal = df[df['Anomaly'] == 1]
    axes[1].scatter(range(len(normal)), normal['Current Price'], 
                   c='blue', alpha=0.6, label='Normal')
    axes[1].scatter(anomalies.index, anomalies['Current Price'], 
                   c='red', alpha=0.8, label='Anomaly', marker='x', s=100)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Current Price')
    axes[1].set_title('Price Data with Anomalies Highlighted')
    axes[1].legend()

plt.tight_layout()
plt.show()
