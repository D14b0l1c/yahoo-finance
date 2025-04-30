# 📂 models/

This folder contains trained machine learning models stored in serialized format (`.pkl`, `.keras`, etc.).

### Files:
- `linear_regression.pkl` – Linear Regression model trained on PE Ratio, Return on Equity, Beta, and EPS.
- `random_forest_regressor.pkl` – Random Forest model trained for robustness and non-linearity.
- `xgboost_regressor.pkl` – XGBoost model optimized for accuracy on structured data.
- `arima_forecast.pkl` – ARIMA model forecasting based on historical stock closing prices.
- `logistic_regression.pkl` – Logistic Regression model classifying next-day movement (up/down).
- `lstm_forecast.keras` – LSTM deep learning model predicting next-day prices using 10-day history.
- `kmeans.pkl` – k-Means clustering model grouping stocks by fundamental similarities.