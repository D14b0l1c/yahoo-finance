# 📂 scripts/

This folder contains training scripts for individual machine learning models.  
Each script loads financial data, trains a specific model, and saves the result to `/models`.

### Files:
- `1_linear_regression.py` – Trains a Linear Regression model.
- `2_random_forest.py` – Trains a Random Forest Regressor.
- `3_xgboost.py` – Trains an XGBoost Regressor.
- `4_arima.py` – Trains an ARIMA model for time series forecasting.
- `5_logistic_regression.py` – Trains a Logistic Regression classifier for next-day price movement.
- `6_lstm.py` – Trains a Keras-based LSTM model using 10-day rolling price windows.