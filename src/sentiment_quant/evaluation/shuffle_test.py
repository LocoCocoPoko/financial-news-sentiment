import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/daily_dataset.csv", parse_dates=["date"])
df = df.sort_values(["Symbol", "date"])
df["ret"] = df.groupby("Symbol")["Adj Close"].pct_change()
for lag in [1, 2, 3, 5]:
    df[f"ret_lag{lag}"] = df.groupby("Symbol")["ret"].shift(lag)
df["vol_5"] = df.groupby("Symbol")["ret"].transform(lambda s: s.rolling(5).std())

feature_cols = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "vol_5"]
df = df.dropna(subset=feature_cols + ["target"])
df["target"] = df["target"].astype(int)

train = df[df["date"].dt.year <= 2022].copy()
test = df[df["date"].dt.year == 2023].copy()

rng = np.random.default_rng(42)
scores = []

for _ in range(200):
    # Shuffle the training labels so they no longer match the features
    shuffled = train.copy()
    shuffled["target"] = rng.permutation(shuffled["target"].values)

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(shuffled[feature_cols], shuffled["target"])
    scores.append(accuracy_score(test["target"], model.predict(test[feature_cols])))

scores = np.array(scores)
print(f"real model accuracy: 0.531")
print(f"scrambled-label accuracy: mean {scores.mean():.3f}, max {scores.max():.3f}")
print(f"share of runs at or above 0.531: {(scores >= 0.531).mean():.3f}")