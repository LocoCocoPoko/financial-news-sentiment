# Financial News Sentiment and Next-Day Stock Direction

Can financial news sentiment improve predictions of whether a stock goes up or down tomorrow? This project tests that for five US stocks, comparing a model that uses only price history with one that also uses FinBERT sentiment scores from news.

**Short answer: not reliably.** Across walk-forward test years 2021 to 2023, both models were only slightly better than guessing, and the news version never clearly beat the price-only version. Full details are in [docs/results.md](docs/results.md).

## Results

Full-article sentiment, walk-forward validation. Each test year was trained only on earlier years.

| Test year | Market accuracy | Market AUC | Sentiment accuracy | Sentiment AUC |
|-----------|-----------------|------------|--------------------|---------------|
| 2021      | 0.517           | 0.522      | 0.522              | 0.520         |
| 2022      | 0.511           | 0.492      | 0.507              | 0.489         |
| 2023      | 0.531           | 0.508      | 0.516              | 0.513         |

AUC measures how well a model ranks up days above down days. 0.5 means no skill. Every AUC here falls between 0.489 and 0.522. Accuracy above 0.5 mostly reflects that the market rose more often than it fell in these years.

Headline-based sentiment (52,303 headlines) gave similar results. See the write-up for those numbers.

## Research question

Does daily news sentiment improve next-day direction predictions compared with price history alone?

## Data

- **Prices:** daily data from Yahoo Finance via `yfinance`, using adjusted close.
- **News:** the FNSPID dataset (`Zihan1004/FNSPID` on Hugging Face), covering 1999 to 2023 for S&P 500 companies. The raw file is 22 GB, so it isn't stored in this repo.
- **Stocks:** KO, MRK, WFC, XOM, GILD. These were chosen because each has news in every year from 2019 to 2023.
  - Dropped: AAPL, MSFT and NVDA (very little news before 2022), JPM and JNJ (no news after 2020), and ETFs and preferred shares (not regular stocks, such as QQQ).
- **Size:** 52,303 headlines, of which 37,934 have full article text. The full-text count is lower because empty bodies were removed.

**Timestamp limitation:** about 99% of article times are placeholders (19:00 or 20:00 New York time). Only the dates are reliable, so the project works at the daily level.

## Key decisions

1. **Daily, not intraday.** Article times can't be trusted, so each trading day is one row.
2. **Predict the next day.** The target is 1 if tomorrow's adjusted close is higher than today's, otherwise 0.
3. **Use yesterday's news.** News features are lagged one trading day, so today's prediction never uses articles that may have appeared after the close.
4. **Roll weekend and holiday news forward.** Saturday news counts toward Monday's features.
5. **Walk-forward validation.** Each test year is trained only on earlier years. No random splits, because those would let the model see the future.
6. **Logistic regression first.** It's simple and easy to inspect. More complex models can come later if the simple ones show a signal.

## Leakage controls

- News features are lagged one trading day (see decision 3).
- The feature scaler is inside the model pipeline, so it learns only from training data.
- Both models are compared on exactly the same test rows.
- An earlier version attached same-day news to same-day prices. It gave 0.528 accuracy, which was inflated by this leak, so it was discarded. The corrected version gave 0.521.
- A shuffle test trained the model on randomly scrambled labels 200 times. None of those runs reached the real model's 2023 accuracy of 0.531 (the highest scrambled run was 0.529).

## Method

- **Market features:** returns lagged 1, 2, 3 and 5 days, plus 5-day rolling volatility.
- **Sentiment features:** daily mean and minimum FinBERT score, and counts of headlines scored above 0.5 or below -0.5. All are lagged one day.
- **FinBERT** (`ProsusAI/finbert`) gives three probabilities per article. The sentiment score is P(positive) minus P(negative), ranging from -1 to +1.
- **Full-article scoring** used only the first 512 word pieces of each article.
- **Model:** logistic regression with standard scaling.

## Project structure

```
src/sentiment_quant/
  data/        price download, news cleaning, article extraction
  nlp/         FinBERT scoring (headlines and full articles)
  features/    daily market and sentiment features
  models/      baseline, market-only, news-volume and sentiment models
  evaluation/  walk-forward and comparison scripts
app/           Streamlit dashboard
docs/          results write-up and decision log
notebooks/     Colab notebook used for full-article scoring
```

## How to reproduce

**Setup**

```bash
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows (use venv/bin/activate on Mac or Linux)
pip install -r requirements.txt
```

**Data**

1. Download `Stock_news/nasdaq_exteral_data.csv` from the FNSPID dataset on Hugging Face and save it as `data/raw/nasdaq_exteral_data.csv`.
2. Filter it to the five stocks:

```bash
python -c "
import pandas as pd
want = ['KO', 'MRK', 'WFC', 'XOM', 'GILD']
parts = []
for chunk in pd.read_csv('data/raw/nasdaq_exteral_data.csv', usecols=['Date', 'Stock_symbol', 'Article_title', 'Url'], chunksize=100000, dtype=str):
    parts.append(chunk[chunk['Stock_symbol'].isin(want)])
pd.concat(parts).to_csv('data/processed/news_5_stocks.csv', index=False)
"
```

**Headline pipeline** (run in order from the project root)

1. `python src/sentiment_quant/data/fetch_prices.py`: downloads prices to `prices.csv`
2. `python src/sentiment_quant/data/clean_news.py`: removes duplicate articles
3. `python src/sentiment_quant/features/build_daily.py`: builds the daily table with the next-day target and article counts
4. `python src/sentiment_quant/nlp/score_news.py`: scores headlines with FinBERT (about an hour on a laptop CPU)
5. `python src/sentiment_quant/features/sentiment_daily.py`: aggregates scores to daily sentiment
6. `python src/sentiment_quant/evaluation/walk_forward.py`: prints the walk-forward table

**Full-article pipeline**

1. `python src/sentiment_quant/data/extract_articles.py`: extracts full text for the five stocks to `news_5_stocks_articles.csv`
2. Upload that file to `notebooks/score_articles_colab.ipynb` on Google Colab with a GPU, and run it to create `news_scored_articles.csv`. Local scoring was too slow on a laptop CPU.
3. Place `news_scored_articles.csv` in `data/processed/`
4. `python src/sentiment_quant/features/sentiment_daily_articles.py`
5. `python src/sentiment_quant/evaluation/walk_forward_articles.py`

**Dashboard**

```bash
streamlit run app/dashboard.py
```

The results table on the dashboard is typed in from `docs/results.md`, so update it there if you rerun the experiments.

## Limitations

- Only headlines, or the first 512 word pieces of each article, are scored.
- Five stocks, one dataset, one prediction horizon, and three test years. The data ends in 2023.
- Article timestamps are mostly placeholders, so intraday or event-timing effects can't be tested.
- Nvidia was excluded early because of data gaps, so recent events such as the November 2026 CEO news fall outside the tested data.
- Missing news is treated as zero sentiment, which may blur the signal.
- Yahoo adjusted prices can be revised after download, so re-running may give slightly different numbers.
- Several model variants were tried. All results are reported, so the write-up doesn't overstate any single one.

## Possible next steps

- An **event study** on specific, dated news events, such as earnings surprises or CEO departures, measuring the stock's move on those days against normal days. This needs reliable event timestamps from another data source.
- Full-article sentiment that reads more of each article, for example by scoring it in chunks.
- More stocks and a longer date range.

## Tech stack

Python, pandas, NumPy, scikit-learn, Hugging Face Transformers, PyTorch, yfinance, Streamlit, Altair, Git, GitHub, Google Colab.

## Disclaimer

This is a research project, not financial advice, and it doesn't execute trades.