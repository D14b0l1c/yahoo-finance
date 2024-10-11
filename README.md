#Stock Ticker Analysis Script (Using Google, Microsoft, and Apple for examples)

This project provides a Python script that uses the `yfinance` library to gather financial data for popular technology companies such as Google (Alphabet Inc.), Microsoft, and Apple. The script retrieves various metrics, including dividend yield, earnings per share (EPS), market capitalization, and other financial indicators. It also calculates the dividend payout ratio where applicable.

## Features
- Fetches detailed stock information such as current price, dividend yield, EPS, P/E ratio, market cap, and more.
- Calculates the dividend payout ratio if dividend and EPS data are available.
- Exports the data to a CSV file for further analysis.

## Prerequisites
To run this script, ensure you have the following Python libraries installed:
- `yfinance`: For accessing stock market data.
- `pandas`: For data handling and exporting to CSV.

You can install these dependencies using:
```bash
pip install yfinance pandas
