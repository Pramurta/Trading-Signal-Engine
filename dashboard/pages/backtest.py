"""Backtest Runner page — run and visualize backtests via the API."""

import os

import httpx
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Backtest Runner", page_icon="📈", layout="wide")
st.title("Backtest Runner")

# --- Sidebar controls ---
symbols = st.sidebar.multiselect(
    "Symbols",
    options=["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA"],
    default=["AAPL", "GOOGL"],
)
days = st.sidebar.slider("Lookback (days)", min_value=30, max_value=500, value=252)
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=10_000_000,
    value=100_000,
    step=10_000,
)
source = st.sidebar.radio("Data Source", options=["synthetic", "yfinance"])

if st.sidebar.button("Run Backtest", type="primary"):
    with st.spinner("Running backtest..."):
        try:
            response = httpx.post(
                f"{API_URL}/backtest/run",
                json={
                    "symbols": symbols,
                    "days": days,
                    "initial_capital": initial_capital,
                    "source": source,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError:
            st.error("Could not connect to API. Is the backend running?")
            st.stop()
        except httpx.HTTPStatusError as e:
            st.error(f"API error: {e.response.text}")
            st.stop()

    metrics = data["metrics"]

    # Performance metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics['total_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col4.metric("Win Rate", f"{metrics['win_rate']:.1%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    col6.metric("Trades", str(metrics["num_trades"]))
    col7.metric("Volatility", f"{metrics['volatility']:.2%}")
    col8.metric("VaR (95%)", f"{metrics['var_95']:.2%}")

    # Equity curve
    st.subheader("Equity Curve")
    equity = data["equity_curve"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=equity,
            mode="lines",
            name="Portfolio Value",
            fill="tozeroy",
            line=dict(color="#00cc96"),
        )
    )
    fig.update_layout(
        height=400,
        template="plotly_dark",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trading Day",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Backtest ID for reference
    st.caption(f"Backtest ID: `{data['backtest_id']}`")
