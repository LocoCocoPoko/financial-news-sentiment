import altair as alt
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

# AUC above 0.5 means better than guessing. Dots show each model's AUC by year.
st.subheader("AUC by year (dashed line = guessing at 0.5)")

# Reshape the table so each row is one model's AUC for one year
long = results.melt(
    id_vars="Test year",
    value_vars=["Market AUC", "Sentiment AUC"],
    var_name="Model",
    value_name="AUC",
)

# Dots instead of bars, with the y-axis zoomed in so small gaps are visible
dots = (
    alt.Chart(long)
    .mark_point(size=150, filled=True)
    .encode(
        x=alt.X("Test year:O"),
        xOffset="Model:N",
        y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.47, 0.55])),
        color="Model:N",
    )
)

# Dashed line at 0.5: a model on this line is no better than guessing
guess_line = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")

st.altair_chart(dots + guess_line, use_container_width=True)

# Part 2: daily sentiment for one stock
st.header("Daily news sentiment for one stock")

daily = pd.read_csv("data/processed/sentiment_daily_articles.csv", parse_dates=["date"])

stock = st.selectbox("Stock", sorted(daily["Symbol"].unique()))
view = daily[daily["Symbol"] == stock].set_index("date")

# Average sentiment per day: above 0 is mostly positive headlines, below 0 mostly negative
st.line_chart(view["sent_mean"])
st.caption("Each point is the average sentiment of that day's articles, from -1 (negative) to +1 (positive).")