import numpy as np
import pandas as pd

NEWS = "data/processed/news_5_stocks_clean.csv"
PRICES = "data/processed/prices.csv"
OUTPUT = "data/processed/daily_dataset.csv"

# --- Prices first: we need the list of trading days ---
prices = pd.read_csv(PRICES, parse_dates=["Date"])
prices = prices.rename(columns={"Date": "date"}).sort_values(["Symbol", "date"])
trading_days = np.sort(prices["date"].unique())

# --- News: normalise dates and keep 2019-2023 ---
news = pd.read_csv(NEWS, dtype=str)
news["date"] = pd.to_datetime(news["Date"], utc=True).dt.tz_localize(None).dt.normalize()
news = news[(news["date"] >= "2019-01-01") & (news["date"] <= "2023-12-31")]

# --- Roll each article forward to the next trading day on or after its date ---
pos = np.searchsorted(trading_days, news["date"].values, side="left")
has_trade_day = pos < len(trading_days)
news = news[has_trade_day].copy()
news["trade_date"] = trading_days[pos[has_trade_day]]

news_daily = (
    news.groupby(["Stock_symbol", "trade_date"])
    .size()
    .rename("n_articles")
    .reset_index()
    .rename(columns={"Stock_symbol": "Symbol", "trade_date": "date"})
)

# --- Target: 1 if tomorrow's Adj Close is higher than today's ---
prices["next_adj_close"] = prices.groupby("Symbol")["Adj Close"].shift(-1)
prices["target"] = (prices["next_adj_close"] > prices["Adj Close"]).astype("Int64")
prices.loc[prices["next_adj_close"].isna(), "target"] = pd.NA

# --- Join: trading days with no news get 0 articles ---
df = prices.merge(news_daily, on=["Symbol", "date"], how="left")
df["n_articles"] = df["n_articles"].fillna(0).astype(int)

df.to_csv(OUTPUT, index=False)
print(df.head())
print(f"{len(df)} rows saved")
print("articles matched:", df["n_articles"].sum())