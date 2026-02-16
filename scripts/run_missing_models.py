"""
Run the 3 models that failed due to missing packages:
- Prophet (Model 9)
- GARCH (Model 10) 
- EBM (Model 8)
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
history_path = os.path.join(BASE_DIR, "data", "portfolio_history.csv")
fundamental_path = os.path.join(BASE_DIR, "data", "portfolio_data.csv")
models_dir = os.path.join(BASE_DIR, "models")

# Load data
history = pd.read_csv(history_path, index_col=0, parse_dates=True)
df = pd.read_csv(fundamental_path)
portfolio_avg = history.mean(axis=1).dropna()

# Features for EBM
feature_cols = ['Market Cap', 'PE Ratio', 'Forward PE', 'Price to Book',
                'Dividend Yield', 'Beta', '52 Week High', '52 Week Low', 
                '50 Day MA', '200 Day MA', 'Volume', 'Avg Volume', 'Profit Margin', 
                'ROE', 'Debt to Equity']
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].copy()
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())
y = df['Current Price'].fillna(df['Current Price'].median())

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 60)
print("Running Previously Failed Models")
print("=" * 60)

# =============================================================================
# MODEL 8: EBM
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
        print(f"    {row['ds'].strftime('%Y-%m-%d')}: ${row['yhat']:.2f}")
    
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

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
