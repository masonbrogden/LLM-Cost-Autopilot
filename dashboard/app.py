import os
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path("data/autopilot.db")

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")

if not DB_PATH.exists():
    st.warning("No benchmark data found. Run the benchmark script to generate data/autopilot.db first.")
    st.stop()

conn = None
try:
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            id,
            created_at,
            request_id,
            routed_tier AS tier,
            prompt_text,
            actual_cost_usd,
            baseline_cost_usd,
            savings_usd,
            quality_score,
            escalated
        FROM requests
        ORDER BY created_at ASC
        """,
        conn,
    )
finally:
    if conn is not None:
        conn.close()

if df.empty:
    st.warning("No benchmark data found. Run the benchmark script to generate data/autopilot.db first.")
    st.stop()

st.title("LLM Cost Autopilot")

metrics = {
    "Total requests": int(len(df)),
    "Total spent": round(float(df["actual_cost_usd"].fillna(0).sum()), 4),
    "Total saved": round(float(df["savings_usd"].fillna(0).sum()), 4),
}
percent_reduction = 0.0
if float(df["baseline_cost_usd"].fillna(0).sum()) > 0:
    percent_reduction = float(
        (1 - (df["actual_cost_usd"].fillna(0).sum() / df["baseline_cost_usd"].fillna(0).sum())) * 100
    )
metrics["Percent reduction"] = round(percent_reduction, 2)

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    if isinstance(value, float):
        if label in {"Total spent", "Total saved"}:
            text = f"${value:,.4f}"
        elif label == "Percent reduction":
            text = f"{value:.2f}%"
        else:
            text = f"{value:.2f}"
    else:
        text = value
    col.metric(label, text)

if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values("created_at")
    df["cum_actual"] = df["actual_cost_usd"].fillna(0).cumsum()
    df["cum_baseline"] = df["baseline_cost_usd"].fillna(0).cumsum()
    st.subheader("Cumulative actual vs baseline cost")
    st.line_chart(df.set_index("created_at")[["cum_actual", "cum_baseline"]])

st.subheader("Request count by tier")
if "tier" in df.columns:
    tier_counts = df["tier"].fillna("unknown").value_counts().sort_index()
    st.bar_chart(tier_counts)

st.subheader("Lowest quality scores")
quality_df = df.dropna(subset=["quality_score"]).copy()
quality_df = quality_df.sort_values("quality_score", ascending=True).head(10)
quality_df["prompt_preview"] = quality_df["prompt_text"].fillna("").str.slice(0, 80)
quality_df = quality_df[["prompt_preview", "tier", "quality_score"]].rename(columns={"tier": "routed_tier", "quality_score": "score"})
st.dataframe(quality_df, use_container_width=True)
