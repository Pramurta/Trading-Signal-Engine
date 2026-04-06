"""Signal Explorer page — visualize trading signals from the API."""

import os

import httpx
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Signal Explorer", page_icon="📊", layout="wide")
st.title("Signal Explorer")

# --- Sidebar controls ---
symbols = st.sidebar.multiselect(
    "Symbols",
    options=["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA"],
    default=["AAPL"],
)
days = st.sidebar.slider("Lookback (days)", min_value=30, max_value=500, value=252)
source = st.sidebar.radio("Data Source", options=["synthetic", "yfinance"])

if st.sidebar.button("Generate Signals", type="primary"):
    with st.spinner("Generating signals..."):
        try:
            response = httpx.post(
                f"{API_URL}/signals/generate",
                json={"symbols": symbols, "days": days, "source": source},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError:
            st.error("Could not connect to API. Is the backend running?")
            st.stop()
        except httpx.HTTPStatusError as e:
            st.error(f"API error: {e.response.text}")
            st.stop()

    for symbol, signals in data["symbols"].items():
        st.subheader(f"{symbol}")

        # Current signal summary
        current = signals["current"]
        cols = st.columns(6)
        cols[0].metric("Z-Score", f"{current['zscore']:+.3f}")
        cols[1].metric("RSI", f"{current['rsi']:+.3f}")
        cols[2].metric("MACD", f"{current['macd']:+.3f}")
        cols[3].metric("Bollinger", f"{current['bollinger']:+.3f}")
        cols[4].metric("Combined", f"{current['combined']:+.3f}")
        cols[5].metric("Direction", current["direction"].upper())

        # Time series chart
        series = signals["series"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

        for name in ["zscore", "rsi", "macd", "bollinger"]:
            fig.add_trace(
                go.Scatter(y=series[name], name=name.upper(), mode="lines"),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                y=series["combined"],
                name="Combined",
                mode="lines",
                line=dict(width=3, color="white"),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(
            height=500,
            title_text=f"{symbol} — Signal History",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
