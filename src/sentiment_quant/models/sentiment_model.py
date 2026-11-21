import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/daily_dataset.csv", parse_dates=["date"])
sent = pd.read_csv("data/processed/sentiment_daily.csv", parse_dates=["date"])

# Attach sentiment to each trading day. Days with no headlines get 0.
df = df.merge(sent, on=["Symbol", "date"], how="left")
news_cols = ["sent_mean", "sent_min", "n_pos", "n_neg"]
df[news_cols] = df[news_cols].fillna(0)
df = df.sort_values(["Symbol", "date"])

# Market features, same as before
df["ret"] = df.groupby("Symbol")["Adj Close"].pct_change()
for lag in [1, 2, 3, 5]:
    df[f"ret_lag{lag}"] = df.groupby("Symbol")["ret"].shift(lag)
df["vol_5"] = df.groupby("Symbol")["ret"].transform(lambda s: s.rolling(5).std())

# Lagged news: each row sees only the previous trading day's headlines
for c in news_cols:
    df[f"{c}_lag1"] = df.groupby("Symbol")[c].shift(1)

feature_cols = (
    ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "vol_5"]
    + [f"{c}_lag1" for c in news_cols]
)
df = df.dropna(subset=feature_cols + ["target"])
df["target"] = df["target"].astype(int)

train = df[df["date"].dt.year <= 2022]
test = df[df["date"].dt.year == 2023]

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(train[feature_cols], train["target"])

pred = model.predict(test[feature_cols])
print(f"sentiment accuracy on 2023: {accuracy_score(test['target'], pred):.3f}")
print(f"share of days predicted up: {pred.mean():.3f}")