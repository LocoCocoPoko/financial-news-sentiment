import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/daily_dataset.csv", parse_dates=["date"])
df = df.sort_values(["Symbol", "date"])
# Use yesterday's news, not today's, because today's articles may come out after the close
df["n_articles"] = df.groupby("Symbol")["n_articles"].shift(1)

# Daily return: today's close vs yesterday's close, per stock
df["ret"] = df.groupby("Symbol")["Adj Close"].pct_change()

# Past returns: shift(lag) moves values down, so each row sees only earlier days
for lag in [1, 2, 3, 5]:
    df[f"ret_lag{lag}"] = df.groupby("Symbol")["ret"].shift(lag)

# Volatility over the last 5 days (how jumpy the stock has been)
df["vol_5"] = df.groupby("Symbol")["ret"].transform(lambda s: s.rolling(5).std())

feature_cols = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "vol_5","n_articles"]
df = df.dropna(subset=feature_cols + ["target"])
df["target"] = df["target"].astype(int)

train = df[df["date"].dt.year <= 2022]
test = df[df["date"].dt.year == 2023]

# The scaler is inside the pipeline, so it learns its numbers from training data only
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(train[feature_cols], train["target"])

pred = model.predict(test[feature_cols])
print(f"news-volume accuracy on 2023: {accuracy_score(test['target'], pred):.3f}")
print(f"share of days predicted up:   {pred.mean():.3f}")