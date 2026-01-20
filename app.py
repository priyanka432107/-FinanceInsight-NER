import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

DATA_PATH = Path("reports/final_output.json")

st.set_page_config(page_title="FinanceInsight AI", layout="wide")

st.title("💹 FinanceInsight AI")
st.caption("End-to-End Financial Intelligence System")

# ---------------- Load Data ----------------
data = json.loads(DATA_PATH.read_text())

# ---------------- Metrics Panel ----------------
st.header("📊 Key Financial Metrics")

metrics = []
tables = []

for x in data:
    if "metric" in x:
        metrics.append(x)
    elif "rows" in x:
        tables.append(x)

df_metrics = pd.DataFrame(metrics)

if not df_metrics.empty:
    st.dataframe(df_metrics, use_container_width=True)

# ---------------- Tables Panel ----------------
st.header("📑 Financial Statements")

tables = [x for x in data if "rows" in x]

for t in tables:
    st.subheader(t["table_type"])
    st.table(pd.DataFrame(t["rows"]))

# ---------------- Charts ----------------
st.header("📈 Visual Analysis")

if not df_metrics.empty:
    chart = px.bar(df_metrics, x="metric", y="value", color="section")
    st.plotly_chart(chart, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built by Akshay • FinanceInsight Project")
