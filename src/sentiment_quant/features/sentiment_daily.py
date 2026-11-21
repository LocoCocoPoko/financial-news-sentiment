import numpy as np
import pandas as pd

NEWS = "data/processed/news_scored.csv"
OUTPUT = "data/processed/sentiment_daily.csv"

news = pd.read_csv(NEWS, dtype={"Stock_symbol": str})
news["date"] = pd.to_datetime(news["Date"], utc=True).dt.tz_localize(None).dt.normalize()
news = news[(news["date"] >= "2019-01-01") & (news["date"] <= "2023-12-31")]

# Roll each article forward to the next trading day, same rule as before
prices = pd.read_csv("data/processed/prices.csv", parse_dates=["Date"])
trading_days = np.sort(prices["Date"].unique())
pos = np.searchsorted(trading_days, news["date"].values, side="left")
keep = pos < len(trading_days)
news = news[keep].copy()
news["trade_date"] = trading_days[pos[keep]]

daily = (
    news.groupby(["Stock_symbol", "trade_date"])
    .agg(
        sent_mean=("sentiment", "mean"),
        sent_min=("sentiment", "min"),
        n_pos=("sentiment", lambda s: (s > 0.5).sum()),
        n_neg=("sentiment", lambda s: (s < -0.5).sum()),
    )
    .reset_index()
    .rename(columns={"Stock_symbol": "Symbol", "trade_date": "date"})
)

daily.to_csv(OUTPUT, index=False)
print(daily.head())
print(len(daily), "stock-days with news")