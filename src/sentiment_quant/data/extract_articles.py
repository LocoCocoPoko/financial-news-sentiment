import pandas as pd

RAW = "data/raw/nasdaq_exteral_data.csv"
OUTPUT = "data/processed/news_5_stocks_articles.csv"
WANT = ["KO", "MRK", "WFC", "XOM", "GILD"]

# The big file is too large to load at once, so read it in chunks of 100,000 rows
pieces = []
for chunk in pd.read_csv(RAW, usecols=["Date", "Stock_symbol", "Article_title", "Url", "Article"], chunksize=100000, dtype=str):
    # Keep only the rows for our 5 stocks
    keep = chunk[chunk["Stock_symbol"].isin(WANT)]
    pieces.append(keep)

# Glue all the kept chunks back into one table
df = pd.concat(pieces)

# Remove rows where the article body is empty
df = df.dropna(subset=["Article"])

# Remove repeats: same link and same stock means the same article
df = df.drop_duplicates(subset=["Url", "Stock_symbol"], keep="first")

df.to_csv(OUTPUT, index=False)
print(len(df), "articles saved")