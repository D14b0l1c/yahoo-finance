import yfinance as yf
import pandas as pd

# List of example stock tickers
tickers = [
    "GOOGL",  # Alphabet Inc. (Google)
    "MSFT",   # Microsoft Corporation
    "AAPL"    # Apple Inc.
]

# Function to fetch and process stock information for a given ticker
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)  # Create a Ticker object using yfinance
    info = stock.info  # Fetch stock information
    dividend_rate = info.get("dividendRate")  # Get the dividend rate
    eps = info.get("trailingEps")  # Get the earnings per share (EPS)
    
    # Calculate Dividend Payout Ratio if EPS is available and non-zero
    dividend_payout_ratio = None
    if eps is not None and eps > 0:  # Avoid division by zero
        dividend_payout_ratio = dividend_rate / eps if dividend_rate else None

    # Return a dictionary with selected stock data
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
        "Dividend Payout Ratio": dividend_payout_ratio  # Newly added field
    }

# Collect data for each ticker and store it in a DataFrame
stock_data = [get_stock_data(ticker) for ticker in tickers]
df = pd.DataFrame(stock_data)

# Save the DataFrame to a CSV file for later use
csv_file = "tech_stocks_data.csv"
df.to_csv(csv_file, index=False)

print(f"Data has been exported to {csv_file}")
