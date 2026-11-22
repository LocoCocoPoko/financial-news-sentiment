import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/daily_dataset.csv", parse_dates=["date"])
sent = pd.read_csv("data/processed/sentiment_daily.csv", parse_dates=["date"])
df = df.merge(sent, on=["Symbol", "date"], how="left")
news_cols = ["sent_mean", "sent_min", "n_pos", "n_neg"]
df[news_cols] = df[news_cols].fillna(0)
df = df.sort_values(["Symbol", "date"])

df["ret"] = df.groupby("Symbol")["Adj Close"].pct_change()
for lag in [1, 2, 3, 5]:
    df[f"ret_lag{lag}"] = df.groupby("Symbol")["ret"].shift(lag)
df["vol_5"] = df.groupby("Symbol")["ret"].transform(lambda s: s.rolling(5).std())
for c in news_cols:
    df[f"{c}_lag1"] = df.groupby("Symbol")[c].shift(1)

market_cols = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "vol_5"]
news_lag_cols = [f"{c}_lag1" for c in news_cols]

# Same rows for both models: drop anything missing from either feature set
df = df.dropna(subset=market_cols + news_lag_cols + ["target"])
df["target"] = df["target"].astype(int)

train = df[df["date"].dt.year <= 2022]
test = df[df["date"].dt.year == 2023]
print(f"test rows used by both models: {len(test)}\n")

def evaluate(name, cols):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(train[cols], train["target"])
    prob = model.predict_proba(test[cols])[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(test["target"], pred)
    auc = roc_auc_score(test["target"], prob)
    print(f"{name:<12} accuracy {acc:.3f}   ROC-AUC {auc:.3f}")

evaluate("market-only", market_cols)
evaluate("sentiment", market_cols + news_lag_cols)