"""
Visualize k-Means Clustering results with a 2D scatter plot.
"""

import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "kmeans.pkl")
data_path = os.path.join(BASE_DIR, "data", "tech_stocks_data.csv")

# Load model and data
kmeans = joblib.load(model_path)
df = pd.read_csv(data_path)
df = df.dropna(subset=["PE Ratio", "Return on Equity", "Beta", "Dividend Yield"])

# Prepare features
features = df[["PE Ratio", "Return on Equity", "Beta", "Dividend Yield"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
df["Cluster"] = kmeans.predict(X_scaled)

# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df["PE Ratio"], df["Return on Equity"], c=df["Cluster"], cmap="viridis", s=50)
plt.title("k-Means Clustering of Stocks")
plt.xlabel("PE Ratio")
plt.ylabel("Return on Equity")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()