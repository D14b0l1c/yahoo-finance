import yfinance as yf
import pandas as pd
import os

# List of example stock tickers
tickers = [
    "GOOGL",  # Alphabet Inc. (Google)
    "MSFT",   # Microsoft Corporation
    "AAPL"    # Apple Inc.
]

# Function to fetch and process stock information for a given ticker
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    dividend_rate = info.get("dividendRate")
    eps = info.get("trailingEps")

    dividend_payout_ratio = None
    if eps is not None and eps > 0:
        dividend_payout_ratio = dividend_rate / eps if dividend_rate else None

    return {
        "Ticker": ticker,
        "Current Price": info.get("currentPrice"),
        "Dividend Yield": info.get("dividendYield"),
        "Forward Dividend": dividend_rate,
        "Ex-Dividend Date": info.get("exDividendDate"),
        "Market Cap": info.get("marketCap"),
        "PE Ratio": info.get("trailingPE"),
        "Beta": info.get("beta"),
        "EPS": eps,
        "Debt-to-Equity": info.get("debtToEquity"),
        "Return on Equity": info.get("returnOnEquity"),
        "Annual Revenue": info.get("totalRevenue"),
        "Cash Flow": info.get("operatingCashflow"),
        "Analyst Recommendations": info.get("recommendationKey"),
        "Dividend Payout Ratio": dividend_payout_ratio
    }

# Collect and store data
stock_data = [get_stock_data(ticker) for ticker in tickers]
df = pd.DataFrame(stock_data)

# Ensure /data directory exists and save CSV there
os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "tech_stocks_data.csv")
df.to_csv(csv_path, index=False)

print(f"Data has been exported to {csv_path}")