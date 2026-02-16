"""
Fetch stock data for portfolio tickers using yfinance.
Creates a comprehensive dataset for all ML models.
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
tickers_path = os.path.join(BASE_DIR, "tickers", "tickers.txt")
output_path = os.path.join(BASE_DIR, "data", "portfolio_data.csv")
history_path = os.path.join(BASE_DIR, "data", "portfolio_history.csv")

# Load tickers
with open(tickers_path, 'r') as f:
    tickers = [line.strip() for line in f if line.strip()]

# Filter out mutual funds that yfinance can't fetch well
problematic = ['VFIAX', 'VMFXX']
tickers = [t for t in tickers if t not in problematic]

print(f"Fetching data for {len(tickers)} tickers...")

# Fetch fundamental data
data_list = []
failed_tickers = []

for i, ticker in enumerate(tickers):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data_list.append({
            'Ticker': ticker,
            'Company Name': info.get('shortName', ticker),
            'Current Price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'Market Cap': info.get('marketCap'),
            'PE Ratio': info.get('trailingPE'),
            'Forward PE': info.get('forwardPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Price to Book': info.get('priceToBook'),
            'Dividend Yield': info.get('dividendYield'),
            'Dividend Rate': info.get('dividendRate'),
            'Beta': info.get('beta'),
            '52 Week High': info.get('fiftyTwoWeekHigh'),
            '52 Week Low': info.get('fiftyTwoWeekLow'),
            '50 Day MA': info.get('fiftyDayAverage'),
            '200 Day MA': info.get('twoHundredDayAverage'),
            'Volume': info.get('volume'),
            'Avg Volume': info.get('averageVolume'),
            'Revenue': info.get('totalRevenue'),
            'Profit Margin': info.get('profitMargins'),
            'ROE': info.get('returnOnEquity'),
            'Debt to Equity': info.get('debtToEquity'),
            'Free Cash Flow': info.get('freeCashflow'),
            'Sector': info.get('sector'),
            'Industry': info.get('industry'),
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tickers)} tickers...")
            
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"  Failed: {ticker} - {str(e)[:50]}")

# Create DataFrame
df = pd.DataFrame(data_list)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nFundamental data saved to: {output_path}")
print(f"Successfully fetched: {len(data_list)} tickers")
if failed_tickers:
    print(f"Failed tickers: {failed_tickers}")

# Fetch historical price data (1 year)
print("\nFetching 1-year historical prices...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Get historical data for all tickers at once
valid_tickers = [t for t in tickers if t not in failed_tickers]
history = yf.download(valid_tickers, start=start_date, end=end_date, progress=True)

# Save Close prices
if 'Close' in history.columns or len(valid_tickers) == 1:
    if len(valid_tickers) == 1:
        close_prices = history[['Close']]
        close_prices.columns = [valid_tickers[0]]
    else:
        close_prices = history['Close']
    close_prices.to_csv(history_path)
    print(f"Historical prices saved to: {history_path}")
    print(f"Date range: {close_prices.index[0].strftime('%Y-%m-%d')} to {close_prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(close_prices)}")

print("\nData fetch complete!")
