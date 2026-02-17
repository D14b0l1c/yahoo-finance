"""
k-Means Clustering
Use for: Unsupervised grouping of stocks based on fundamentals
Input: PE Ratio, ROE, Beta, Dividend Yield
Output: Cluster labels (e.g., growth vs. value)
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "research", "portfolio_data.csv")
model_path = os.path.join(BASE_DIR, "models", "kmeans.pkl")

# Load data
df = pd.read_csv(data_path)
df = df.dropna(subset=["PE Ratio", "Return on Equity", "Beta", "Dividend Yield"])

# Select features for clustering
features = df[["PE Ratio", "Return on Equity", "Beta", "Dividend Yield"]]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Fit k-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(kmeans, model_path)

print(f"k-Means clustering complete. Saved model to: {model_path}")
print(df[["Ticker", "PE Ratio", "Return on Equity", "Beta", "Dividend Yield", "Cluster"]].head())
