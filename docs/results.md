# Results

## Question
Does FinBERT sentiment from financial news headlines improve next-day direction
predictions for five US stocks, compared with using price history alone?

## Setup
- Stocks: KO, MRK, WFC, XOM, GILD
- News: FNSPID dataset, headlines only, 2019 to 2023
- Target: 1 if tomorrow's adjusted close is higher than today's, else 0
- Market features: lagged returns (1, 2, 3, 5 days) and 5-day volatility
- Sentiment features: daily mean and minimum headline score, counts of strongly
  positive and strongly negative headlines, all lagged one trading day
- Model: logistic regression with standard scaling
- Validation: walk-forward. Each test year is trained only on earlier years.

## Leakage fix
The first version attached each day's headlines to that day's row. Some of those
articles may have been published after the market close, so the model could have
used information it would not have had. We now lag all news features by one trading
day. The earlier news-volume result (0.528) was inflated by this and is discarded.

## Walk-forward results

| Test year | Market accuracy | Market AUC | Sentiment accuracy | Sentiment AUC |
|-----------|-----------------|------------|--------------------|---------------|
| 2021      | 0.517           | 0.522      | 0.530              | 0.528         |
| 2022      | 0.511           | 0.492      | 0.502              | 0.495         |
| 2023      | 0.531           | 0.508      | 0.521              | 0.509         |

## Interpretation
- Sentiment's AUC is slightly higher than market-only's in all three years, but
  every gap is 0.006 or less.
- All AUC values fall between 0.49 and 0.53. A value of 0.5 is random guessing.
- Accuracy above 0.5 mostly reflects that the market rose more often than it fell
  in these years, not skill in ranking days.
- Conclusion: in this setup, headline sentiment does not reliably improve
  next-day direction predictions.

## Limitations
- Headlines only. Full article text may carry more information.
- Five stocks, one dataset, one prediction horizon.
- FNSPID news timestamps are mostly placeholders (about 99% fall at 19:00 or
  20:00 New York time), so intraday timing could not be tested.
- FNSPID ends in 2023, so there is no 2024 test year.
- Yahoo adjusted prices can be revised after download, so results may change
  slightly if the data is re-downloaded.
- Only three test years. 2020 could not be tested because the training window
  would be just one year.
- Several model variants were tried. Choosing the best one would overstate results,
  so this write-up reports all of them.

## Full-article sentiment

We repeated the walk-forward test using FinBERT scores from the full article
text instead of the headline. Articles were scored on Google Colab (GPU).
The dataset contained 37,934 articles with body text after removing empty and
duplicate rows, compared with 52,303 headlines. Because of this, the two sets of
results cover slightly different news.

| Test year | Market accuracy | Market AUC | Sentiment accuracy | Sentiment AUC |
|-----------|-----------------|------------|--------------------|---------------|
| 2021      | 0.517           | 0.522      | 0.522              | 0.520         |
| 2022      | 0.511           | 0.492      | 0.507              | 0.489         |
| 2023      | 0.531           | 0.508      | 0.516              | 0.513         |

Using full text did not change the conclusion. Sentiment is below market-only
in 2021 and 2022, and only slightly above it on AUC in 2023 (0.513 vs 0.508).
FinBERT reads only the first 512 word pieces of each article, so it sees roughly
the opening few paragraphs.