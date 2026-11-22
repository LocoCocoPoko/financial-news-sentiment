import pandas as pd
import streamlit as st

st.title("Financial News Sentiment: Results")

# Part 1: walk-forward results (full-article version)
# These numbers are copied from the walk-forward run in docs/results.md.
# If you rerun the experiments, update them here.
st.header("Walk-forward results")

results = pd.DataFrame({
    "Test year": [2021, 2022, 2023],
    "Market accuracy": [0.517, 0.511, 0.531],
    "Market AUC": [0.522, 0.492, 0.508],
    "Sentiment accuracy": [0.522, 0.507, 0.516],
    "Sentiment AUC": [0.520, 0.489, 0.513],
})

st.dataframe(results, hide_index=True)

# AUC above 0.5 means better than guessing. Bars show each model's AUC by year.
st.subheader("AUC by year (0.5 = guessing)")
st.bar_chart(results.set_index("Test year")[["Market AUC", "Sentiment AUC"]])

# Part 2: daily sentiment for one stock
st.header("Daily news sentiment for one stock")

daily = pd.read_csv("data/processed/sentiment_daily_articles.csv", parse_dates=["date"])

stock = st.selectbox("Stock", sorted(daily["Symbol"].unique()))
view = daily[daily["Symbol"] == stock].set_index("date")

# Average sentiment per day: above 0 is mostly positive headlines, below 0 mostly negative
st.line_chart(view["sent_mean"])
st.caption("Each point is the average sentiment of that day's articles, from -1 (negative) to +1 (positive).")