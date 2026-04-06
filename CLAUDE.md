# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python framework for real-time market data processing, statistical signal generation, and risk-managed trading strategy development. Uses async I/O for concurrent data fetching, vectorized NumPy operations for signal computation, and supports both synthetic data (Geometric Brownian Motion) and real market data via yfinance.

## Commands

```bash
# Install dependencies (use the existing .venv)
pip install -r requirements.txt

# Run all tests
python -m pytest src/tests/ -v

# Run a single test file
python -m pytest src/tests/test_signals.py -v

# Run a single test by name
python -m pytest src/tests/test_signals.py -v -k "test_name"

# Run with coverage
python -m pytest src/tests/ -v --cov=src

# Run the backtest demo
python run_backtest.py

# Type checking
mypy src/
```

## Architecture

The system is a pipeline: **Data Handler -> Signal Engine -> Risk Manager -> Strategy -> Backtester**.

- **`src/data_handler.py`** — Async market data fetching. `MarketDataHandler` supports two `DataSource` modes: `SYNTHETIC` (GBM-generated, no internet) and `YFINANCE` (real Yahoo Finance data via thread pool executor). All data flows through the `MarketData` dataclass (OHLCV arrays). yfinance calls are wrapped with `run_in_executor` to stay non-blocking.

- **`src/signals.py`** — `SignalEngine` computes four indicators (Z-Score, RSI, Bollinger Bands, MACD), all returning normalized `Signal` objects in [-1, 1]. `combine_signals()` merges them with configurable weights. Helper methods (`_rolling_mean`, `_rolling_std`, `_ema`) are vectorized with NumPy.

- **`src/risk_manager.py`** — `RiskManager` handles position sizing via Kelly Criterion (fractional, default quarter-Kelly), volatility scaling, and drawdown-based risk reduction. `Portfolio` dataclass tracks capital, positions, and peak equity. `calculate_portfolio_risk()` computes Sharpe, VaR, max drawdown, and assigns a `RiskLevel`.

- **`src/strategy.py`** — `TradingStrategy` orchestrates the pipeline: fetches data, generates/combines signals, applies risk management, and produces `TradeDecision` objects. `run_backtest()` simulates day-by-day with transaction costs, tracking positions and PnL per symbol.

- **`src/backtest.py`** — `Backtester` is a standalone vectorized backtesting engine operating on (T x N) price/signal matrices. Supports walk-forward analysis and parameter sweeps. Separate from Strategy's built-in backtest — this one is fully vectorized with no per-day loops.

- **`run_backtest.py`** — Top-level demo script that wires all components together and prints formatted results.

## Key Patterns

- Tests live in `src/tests/`, not a top-level `tests/` directory. Imports use `from src.module import ...`.
- Async tests use `asyncio.run()` inside regular test methods (not pytest-asyncio markers).
- The `DataSource` enum controls whether `MarketDataHandler` uses synthetic or real data — synthetic is the default and requires no network.
- All signal values are normalized to [-1, 1] and inverted for mean-reversion logic (negative z-score = positive/buy signal).
- The `Backtester` class in `backtest.py` and the `run_backtest()` method in `strategy.py` are two distinct backtesting approaches: vectorized matrix ops vs. day-by-day simulation.
