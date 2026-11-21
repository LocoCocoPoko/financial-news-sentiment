import pandas as pd
from finbert import score_texts

INPUT = "data/processed/news_5_stocks_clean.csv"
OUTPUT = "data/processed/news_scored.csv"
BATCH = 64

news = pd.read_csv(INPUT, dtype=str)
titles = news["Article_title"].fillna("").tolist()

rows = []
for i in range(0, len(titles), BATCH):
    rows.extend(score_texts(titles[i:i + BATCH]))
    if (i // BATCH) % 50 == 0:
        print(f"{i + BATCH} / {len(titles)} done", flush=True)

scores = pd.DataFrame(rows)
out = pd.concat([news.reset_index(drop=True), scores], axis=1)

# Sentiment score: how much more positive than negative the headline is
out["sentiment"] = out["p_positive"] - out["p_negative"]

out.to_csv(OUTPUT, index=False)
print("saved", len(out), "rows")