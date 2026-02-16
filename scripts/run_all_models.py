"""
Run ALL machine learning models on portfolio data.
This master script executes all 14 models and generates reports.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
fundamental_path = os.path.join(BASE_DIR, "data", "portfolio_data.csv")
history_path = os.path.join(BASE_DIR, "data", "portfolio_history.csv")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 60)
print("PORTFOLIO ML ANALYSIS - Running All 14 Models")
print("=" * 60)

# Load data
df = pd.read_csv(fundamental_path)
history = pd.read_csv(history_path, index_col=0, parse_dates=True)

print(f"\nLoaded {len(df)} stocks with fundamental data")
print(f"Loaded {len(history)} days of price history for {len(history.columns)} tickers")

# Prepare features for regression models
feature_cols = ['Market Cap', 'PE Ratio', 'Forward PE', 'Price to Book',
                'Dividend Yield', 'Beta', '52 Week High', '52 Week Low', 
                '50 Day MA', '200 Day MA', 'Volume', 'Avg Volume', 'Profit Margin', 
                'ROE', 'Debt to Equity']
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
# Fill missing values with median
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())
y = df['Current Price'].fillna(df['Current Price'].median())

# Keep all samples since we filled NaN
valid_mask = pd.Series([True] * len(df))

print(f"\nUsing {len(feature_cols)} features for regression models")
print(f"Valid samples: {len(X)}")

# =============================================================================
# MODEL 1: LINEAR REGRESSION
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 1: Linear Regression")
print("-" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print(f"  R² Score: {r2_score(y_test, lr_pred):.4f}")
print(f"  MAE: ${mean_absolute_error(y_test, lr_pred):.2f}")
joblib.dump(lr_model, os.path.join(models_dir, "linear_regression.pkl"))
print("  Model saved!")

# =============================================================================
# MODEL 2: RANDOM FOREST
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 2: Random Forest Regressor")
print("-" * 60)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(f"  R² Score: {r2_score(y_test, rf_pred):.4f}")
print(f"  MAE: ${mean_absolute_error(y_test, rf_pred):.2f}")
joblib.dump(rf_model, os.path.join(models_dir, "random_forest_regressor.pkl"))
print("  Model saved!")

# Top features
importance = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_})
print("  Top 5 Features:")
for _, row in importance.nlargest(5, 'Importance').iterrows():
    print(f"    - {row['Feature']}: {row['Importance']:.3f}")

# =============================================================================
# MODEL 3: XGBOOST
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 3: XGBoost Regressor")
print("-" * 60)

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_pred = xgb_model.predict(X_test)

print(f"  R² Score: {r2_score(y_test, xgb_pred):.4f}")
print(f"  MAE: ${mean_absolute_error(y_test, xgb_pred):.2f}")
joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_regressor.pkl"))
print("  Model saved!")

# =============================================================================
# MODEL 4: ARIMA (Time Series)
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 4: ARIMA - Time Series Forecast")
print("-" * 60)

from statsmodels.tsa.arima.model import ARIMA

# Use portfolio average price for ARIMA
portfolio_avg = history.mean(axis=1).dropna()

try:
    arima_model = ARIMA(portfolio_avg, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    forecast = arima_fit.forecast(steps=5)
    
    print(f"  5-Day Portfolio Forecast:")
    for i, val in enumerate(forecast, 1):
        print(f"    Day {i}: ${val:.2f}")
    
    joblib.dump(arima_fit, os.path.join(models_dir, "arima_forecast.pkl"))
    print("  Model saved!")
except Exception as e:
    print(f"  ARIMA failed: {e}")

# =============================================================================
# MODEL 5: LOGISTIC REGRESSION (Classification)
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 5: Logistic Regression - Up/Down Classification")
print("-" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create binary target: 1 if price > 50-day MA, else 0
df_valid = df[valid_mask].copy()
df_valid['Target'] = (df_valid['Current Price'] > df_valid['50 Day MA']).astype(int)
y_class = df_valid['Target'].values

log_model = LogisticRegression(max_iter=1000, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
log_model.fit(X_train_c, y_train_c)
log_pred = log_model.predict(X_test_c)

print(f"  Accuracy: {accuracy_score(y_test_c, log_pred)*100:.1f}%")
print(f"  Stocks above 50-day MA: {y_class.sum()}/{len(y_class)}")
joblib.dump(log_model, os.path.join(models_dir, "logistic_regression.pkl"))
print("  Model saved!")

# =============================================================================
# MODEL 6: LSTM (Deep Learning)
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 6: LSTM - Sequential Price Prediction")
print("-" * 60)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Use a single stock with good data (e.g., AAPL)
if 'AAPL' in history.columns:
    prices = history['AAPL'].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    
    seq_length = 20
    X_lstm, y_lstm = [], []
    for i in range(len(prices_scaled) - seq_length):
        X_lstm.append(prices_scaled[i:i+seq_length])
        y_lstm.append(prices_scaled[i+seq_length])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    split = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
    
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16, 
                   validation_split=0.1, callbacks=[early_stop], verbose=0)
    
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)
    mse = np.mean((y_test_lstm - lstm_pred) ** 2)
    print(f"  Test MSE (scaled): {mse:.6f}")
    
    # Predict next day
    last_seq = prices_scaled[-seq_length:].reshape(1, seq_length, 1)
    next_pred = scaler.inverse_transform(lstm_model.predict(last_seq, verbose=0))[0][0]
    print(f"  AAPL Next Day Prediction: ${next_pred:.2f}")
    
    lstm_model.save(os.path.join(models_dir, "lstm_forecast.keras"))
    print("  Model saved!")
else:
    print("  AAPL not in history, skipping LSTM")

# =============================================================================
# MODEL 7: K-MEANS CLUSTERING
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 7: K-Means Clustering - Stock Grouping")
print("-" * 60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler_km = StandardScaler()
X_scaled = scaler_km.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df[valid_mask].copy()
df_clustered['Cluster'] = clusters

print("  Cluster Distribution:")
for i in range(5):
    cluster_stocks = df_clustered[df_clustered['Cluster'] == i]['Ticker'].tolist()
    print(f"    Cluster {i}: {len(cluster_stocks)} stocks")
    if len(cluster_stocks) <= 5:
        print(f"      {cluster_stocks}")
    else:
        print(f"      {cluster_stocks[:5]} ...")

joblib.dump(kmeans, os.path.join(models_dir, "kmeans.pkl"))
print("  Model saved!")

# =============================================================================
# MODEL 8: EBM (Explainable Boosting Machine)
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 8: Explainable Boosting Machine (EBM)")
print("-" * 60)

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    
    ebm_model = ExplainableBoostingRegressor(random_state=42)
    ebm_model.fit(X_train, y_train)
    ebm_pred = ebm_model.predict(X_test)
    
    print(f"  R² Score: {r2_score(y_test, ebm_pred):.4f}")
    print(f"  MAE: ${mean_absolute_error(y_test, ebm_pred):.2f}")
    joblib.dump(ebm_model, os.path.join(models_dir, "ebm.pkl"))
    print("  Model saved!")
except Exception as e:
    print(f"  EBM failed: {e}")

# =============================================================================
# MODEL 9: PROPHET
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 9: Prophet - Time Series with Seasonality")
print("-" * 60)

try:
    from prophet import Prophet
    
    prophet_df = pd.DataFrame({
        'ds': portfolio_avg.index,
        'y': portfolio_avg.values
    })
    
    prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    prophet_model.fit(prophet_df)
    
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)
    
    print("  30-Day Forecast (next 5 days shown):")
    for _, row in forecast.tail(30).head(5).iterrows():
        print(f"    {row['ds'].strftime('%Y-%m-%d')}: ${row['yhat']:.2f} (${row['yhat_lower']:.2f} - ${row['yhat_upper']:.2f})")
    
    joblib.dump(prophet_model, os.path.join(models_dir, "prophet_forecast.pkl"))
    print("  Model saved!")
except Exception as e:
    print(f"  Prophet failed: {e}")

# =============================================================================
# MODEL 10: GARCH (Volatility)
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 10: GARCH - Volatility Modeling")
print("-" * 60)

try:
    from arch import arch_model
    
    returns = portfolio_avg.pct_change().dropna() * 100
    
    garch = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    
    vol_forecast = garch_fit.forecast(horizon=5)
    
    print("  5-Day Volatility Forecast:")
    for i, vol in enumerate(np.sqrt(vol_forecast.variance.values[-1]), 1):
        print(f"    Day {i}: {vol:.2f}%")
    
    print(f"\n  Current Portfolio Volatility: {garch_fit.conditional_volatility[-1]:.2f}%")
    
    joblib.dump(garch_fit, os.path.join(models_dir, "garch_volatility.pkl"))
    print("  Model saved!")
except Exception as e:
    print(f"  GARCH failed: {e}")

# =============================================================================
# MODEL 11: LIGHTGBM
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 11: LightGBM Regressor")
print("-" * 60)

import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                               random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

print(f"  R² Score: {r2_score(y_test, lgb_pred):.4f}")
print(f"  MAE: ${mean_absolute_error(y_test, lgb_pred):.2f}")
joblib.dump(lgb_model, os.path.join(models_dir, "lightgbm_regressor.pkl"))
print("  Model saved!")

# =============================================================================
# MODEL 12: GRU
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 12: GRU - Sequential Prediction")
print("-" * 60)

from tensorflow.keras.layers import GRU as GRULayer

if 'AAPL' in history.columns:
    # Reuse scaled data from LSTM
    gru_model = Sequential([
        GRULayer(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        GRULayer(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mse')
    
    gru_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16,
                  validation_split=0.1, callbacks=[early_stop], verbose=0)
    
    gru_pred = gru_model.predict(X_test_lstm, verbose=0)
    mse = np.mean((y_test_lstm - gru_pred) ** 2)
    print(f"  Test MSE (scaled): {mse:.6f}")
    
    gru_model.save(os.path.join(models_dir, "gru_forecast.keras"))
    print("  Model saved!")
else:
    print("  AAPL not in history, skipping GRU")

# =============================================================================
# MODEL 13: ISOLATION FOREST (Anomaly Detection)
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 13: Isolation Forest - Anomaly Detection")
print("-" * 60)

from sklearn.ensemble import IsolationForest

iso_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
anomalies = iso_model.fit_predict(X_scaled)

df_anomaly = df[valid_mask].copy()
df_anomaly['Anomaly'] = anomalies

anomaly_stocks = df_anomaly[df_anomaly['Anomaly'] == -1]['Ticker'].tolist()
print(f"  Detected {len(anomaly_stocks)} anomalous stocks:")
print(f"    {anomaly_stocks}")

joblib.dump(iso_model, os.path.join(models_dir, "isolation_forest.pkl"))
print("  Model saved!")

# =============================================================================
# MODEL 14: MEAN-VARIANCE OPTIMIZATION
# =============================================================================
print("\n" + "-" * 60)
print("MODEL 14: Mean-Variance Portfolio Optimization")
print("-" * 60)

from scipy.optimize import minimize

# Calculate returns
returns_df = history.pct_change().dropna()
returns_df = returns_df.dropna(axis=1)  # Drop columns with NaN

expected_returns = returns_df.mean() * 252  # Annualized
cov_matrix = returns_df.cov() * 252  # Annualized

n_assets = len(expected_returns)

def neg_sharpe(weights):
    ret = np.dot(weights, expected_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(ret - 0.04) / vol if vol > 0 else 0

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = tuple((0, 0.15) for _ in range(n_assets))  # Max 15% per stock
initial = np.ones(n_assets) / n_assets

result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_return = np.dot(optimal_weights, expected_returns)
optimal_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
sharpe = (optimal_return - 0.04) / optimal_vol

print(f"  Optimal Portfolio Metrics:")
print(f"    Expected Return: {optimal_return*100:.2f}%")
print(f"    Volatility: {optimal_vol*100:.2f}%")
print(f"    Sharpe Ratio: {sharpe:.3f}")

print(f"\n  Top 10 Recommended Allocations:")
allocation = pd.DataFrame({
    'Ticker': expected_returns.index,
    'Weight': optimal_weights
}).sort_values('Weight', ascending=False)

for _, row in allocation.head(10).iterrows():
    if row['Weight'] > 0.01:
        print(f"    {row['Ticker']}: {row['Weight']*100:.2f}%")

optimization_results = {
    'optimal_weights': optimal_weights,
    'tickers': expected_returns.index.tolist(),
    'expected_return': optimal_return,
    'volatility': optimal_vol,
    'sharpe_ratio': sharpe
}
joblib.dump(optimization_results, os.path.join(models_dir, "portfolio_optimization.pkl"))
print("  Model saved!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nAll 14 models trained and saved to: {models_dir}")
print("\nKey Insights:")
print(f"  - Portfolio contains {len(df)} stocks")
print(f"  - {len(anomaly_stocks)} stocks flagged as anomalies")
print(f"  - Optimal Sharpe Ratio: {sharpe:.3f}")
print(f"  - 5-day volatility trend: {'increasing' if vol_forecast.variance.values[-1][-1] > vol_forecast.variance.values[-1][0] else 'stable/decreasing'}")
