# Decision Log

## 1. News source
FNSPID (nasdaq_exteral_data.csv), a free academic dataset covering 1999-2023.
Stored in data/raw/ (not in GitHub).

## 2. Timestamps
News dates have no reliable time of day. About 99% of rows sit at placeholder
times (20:00 or 19:00 New York time). So we cannot tell whether an article came
before or after the 4pm close.

## 3. Prediction setup (Option A: daily lag)
- Features use news dated on day T and market data up to the close of day T.
- Target: whether the close on day T+1 is higher than the close on day T.
- Limitation: news is one day older than ideal, so any sentiment effect may be
  weaker. This goes in the final write-up.

## 4. Stock universe
KO, MRK, WFC, XOM, GILD. Each has news in every year from 2019 to 2023.
Dropped AAPL, MSFT, NVDA (almost no news before 2022), JPM and JNJ (no news after 2020).
ETFs and preferred shares excluded (e.g. QQQ, BROGW).

## 5. Train/test split
Train: 2019-2022. Test: 2023. The dataset has no 2024 news.

## 6. Filtered data
data/processed/news_5_stocks.csv: 52,318 articles, 15 duplicate article+stock pairs
(to be removed in Phase 2).