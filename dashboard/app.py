"""Trading Signal Engine — Streamlit Dashboard."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Trading Signal Engine",
    page_icon="📈",
    layout="wide",
)

st.title("Trading Signal Engine")
st.markdown("Statistical signal generation and backtesting dashboard.")

# API connection status
try:
    response = httpx.get(f"{API_URL}/health", timeout=5.0)
    if response.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Error")
except httpx.ConnectError:
    st.sidebar.error("API Unreachable")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("pages/signals.py", label="Signal Explorer", icon="📊")
st.sidebar.page_link("pages/backtest.py", label="Backtest Runner", icon="📈")
