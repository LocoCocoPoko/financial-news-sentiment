import yfinance as yf

TICKERS = ["KO", "MRK", "WFC", "XOM", "GILD"]
START = "2019-01-01"
END = "2024-01-01"  # end date is excluded, so this covers all of 2023
OUTPUT = "data/processed/prices.csv"

raw = yf.download(TICKERS, start=START, end=END, auto_adjust=False, progress=False)

# Keep the two price columns, then reshape to one row per stock per day
prices = raw[["Adj Close", "Close"]].stack(level=1).reset_index()
prices.columns = ["Date", "Symbol", "Adj Close", "Close"]

prices.to_csv(OUTPUT, index=False)
print(prices.head())
print(f"{len(prices)} rows saved")