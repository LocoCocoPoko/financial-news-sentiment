import pandas as pd

df = pd.read_csv("data/processed/daily_dataset.csv", parse_dates=["date"])
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

# Chronological split: train on 2019-2022, test on 2023
train = df[df["date"].dt.year <= 2022]
test = df[df["date"].dt.year == 2023]

print("train rows:", len(train), "| test rows:", len(test))

# Baseline 1: always predict "up"
always_up = (test["target"] == 1).mean()
# Baseline 2: always predict "down"
always_down = (test["target"] == 0).mean()

print(f"always-up accuracy on 2023:   {always_up:.3f}")
print(f"always-down accuracy on 2023: {always_down:.3f}")