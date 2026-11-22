import pandas as pd
from finbert import score_texts

INPUT = "data/processed/news_5_stocks_articles.csv"
OUTPUT = "data/processed/news_scored_articles.csv"

# Read the articles. dtype=str keeps everything as plain text.
news = pd.read_csv(INPUT, dtype=str)

# Test mode: keep only the first 200 articles.
# Remove the # from the next line for the test, put it back for the full run.
# news = news.head(200)

# Make a list with one article body per entry
texts = news["Article"].fillna("").tolist()

# FinBERT works on small groups, so we feed it 16 articles at a time.
# Scores come back in the same order as the articles.
all_scores = []
for start in range(0, len(texts), 16):
    group = texts[start:start + 16]
    all_scores.extend(score_texts(group))
    if start % 1600 == 0:
        print(start, "of", len(texts), "done", flush=True)

# Turn the list of scores into a table, then place it next to the articles
scores = pd.DataFrame(all_scores)
out = pd.concat([news.reset_index(drop=True), scores], axis=1)

# One summary number per article: how much more positive than negative it is
out["sentiment"] = out["p_positive"] - out["p_negative"]

out.to_csv(OUTPUT, index=False)
print("saved", len(out), "rows")