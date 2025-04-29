# 📂 visualize/

This folder contains scripts to inspect and evaluate the predictions made by trained models.  
Each script loads a `.pkl` or `.keras` model, runs it against the dataset, and prints a comparison of actual vs. predicted prices or movements.

### Files:
- `visualize_1_linear_regression.py` – Visualizes output from the Linear Regression model.
- `visualize_2_random_forest.py` – Visualizes output from the Random Forest Regressor.
- `visualize_3_xgboost.py` – Visualizes output from the XGBoost Regressor.
- `visualize_4_arima.py` – Forecasts future stock prices using the ARIMA model.
- `visualize_5_logistic_regression.py` – Predicts and compares up/down movements using Logistic Regression.
- `visualize_6_lstm.py` – Predicts next-day price using LSTM memory-based modeling.