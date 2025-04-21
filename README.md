# 📈 Yahoo Finance Predictive Modeling

This project demonstrates how to use machine learning to analyze financial data pulled from Yahoo Finance using `yfinance`.

---

## ✅ Implemented Models

### 🧮 1. Linear Regression
- Predicts next-day stock price using: PE Ratio, Return on Equity, Beta, and EPS
- Model file: `models/linear_regression.pkl`
- Script: `scripts/1_linear_regression.py`
- Visualizer: `visualize/visualize_1_linear_regression.py`

### 🌲 2. Random Forest Regressor
- Predicts next-day stock price using: PE Ratio, Return on Equity, Beta, and EPS
- Model file: `models/random_forest_regressor.pkl`
- Script: `scripts/2_random_forest.py`
- Visualizer: `visualize/visualize_2_random_forest.py`

### ⚡ 3. XGBoost Regressor
- Predicts next-day stock price using: PE Ratio, Return on Equity, Beta, and EPS
- Model file: `models/xgboost_regressor.pkl`
- Script: `scripts/3_xgboost.py`
- Visualizer: `visualize/visualize_3_xgboost.py`

---

## 📁 Project Structure

```
yahoo-finance/
├── data/               # Source CSVs and raw data pulls
├── models/             # Trained model files (.pkl, .keras)
├── notebooks/          # Interactive EDA and prediction outputs
├── scripts/            # One-off ML model scripts for training
├── visualize/          # Code to inspect predictions from saved models
├── requirements.txt    # Install dependencies
└── README.md           # This file
```

---

## 📚 Data Source
All data comes from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` API.

---

## 🔮 What's Next

Each of these models will be added to the project in the same format:
- One training script (`/scripts/`)
- One trained model (`/models/`)
- One visualizer (`/visualize/`)
- One README update

### Planned Additions

- 📉 ARIMA (Time Series Forecasting)
- 🧠 LSTM (Deep Learning for sequences)
- 📊 Logistic Regression (Directional classification)
- 🔍 k-Means (Clustering for segmentation)
- 📈 EBM (Explainable Boosting Machine for interpretability)