import pandas as pd

INPUT = "data/processed/news_5_stocks.csv"
OUTPUT = "data/processed/news_5_stocks_clean.csv"

df = pd.read_csv(INPUT, dtype=str)
before = len(df)

# Sort by date so that "keep first" keeps the earliest date
df["ts"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("ts")
df = df.drop_duplicates(subset=["Url", "Stock_symbol"], keep="first")
df = df.drop(columns=["ts"])

df.to_csv(OUTPUT, index=False)
print(f"Before: {before}, After: {len(df)}, Removed: {before - len(df)}")