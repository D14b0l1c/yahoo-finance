# 📈 Yahoo Finance Predictive Modeling

This project demonstrates how to use machine learning to analyze financial data pulled from Yahoo Finance using `yfinance`.

---

## ✅ Implemented Models

### 🧮 1. Linear Regression
- Predicts next-day stock price using: PE Ratio, Return on Equity, Beta, and EPS
- Model file: `models/linear_regression.pkl`
- Script: `scripts/1_linear_regression.py`
- Visualizer: `visualize/visualize_1_linear_regression.py`

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
Upcoming models:
- Random Forest
- XGBoost
- ARIMA
- LSTM