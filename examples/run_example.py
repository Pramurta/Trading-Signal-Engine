#!/usr/bin/env python3
"""Fetch 1 year of AAPL data, run a backtest, save results and equity curve."""

import asyncio
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import BacktestConfig, Backtester
from src.data_handler import DataSource, MarketDataHandler
from src.signals import SignalEngine

SYMBOLS = ["AAPL"]
DAYS = 252
OUTPUT_DIR = Path(__file__).resolve().parent


async def main():
    # Fetch real data
    print(f"Fetching {DAYS} days of data for {SYMBOLS} via yfinance...")
    handler = MarketDataHandler(source=DataSource.YFINANCE)
    data = await handler.stream_market_data(SYMBOLS, days=DAYS)

    if not data:
        print("ERROR: No data returned. Check your internet connection.")
        return

    engine = SignalEngine()
    lines = []

    for symbol, md in data.items():
        prices = md.close
        returns = np.diff(np.log(prices))
        vol = np.std(returns) * np.sqrt(252)
        total_ret = (prices[-1] / prices[0]) - 1
        lines.append(
            f"{symbol}: {len(md)} days, "
            f"${prices[-1]:.2f}, "
            f"vol={vol:.1%}, return={total_ret:+.1%}"
        )

    # Generate signals
    min_len = min(len(d.close) for d in data.values())
    prices_matrix = np.column_stack([d.close[:min_len] for d in data.values()])
    signals_matrix = np.column_stack(
        [
            engine.combine_signals(
                {
                    "zscore": engine.calculate_zscore(d.close[:min_len]),
                    "rsi": engine.calculate_rsi(d.close[:min_len]),
                    "macd": engine.calculate_macd(d.close[:min_len]),
                    "bollinger": engine.calculate_bollinger_bands(d.close[:min_len])[0],
                },
                weights={
                    "zscore": 0.35,
                    "rsi": 0.25,
                    "macd": 0.25,
                    "bollinger": 0.15,
                },
            ).values
            for d in data.values()
        ]
    )

    # Backtest
    config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=10,
        slippage_bps=5,
        max_position_weight=0.20,
    )
    backtester = Backtester(config)
    metrics, equity, _ = backtester.run(prices_matrix, signals_matrix)

    # Build output text
    lines.append("")
    lines.append(f"Backtest: {min_len} days, ${config.initial_capital:,.0f}")
    lines.append(f"Final equity: ${equity[-1]:,.0f}")
    lines.append("")
    lines.append(f"Total Return:     {metrics.total_return:>10.2%}")
    lines.append(f"CAGR:             {metrics.cagr:>10.2%}")
    lines.append(f"Sharpe Ratio:     {metrics.sharpe_ratio:>10.2f}")
    lines.append(f"Sortino Ratio:    {metrics.sortino_ratio:>10.2f}")
    lines.append(f"Max Drawdown:     {metrics.max_drawdown:>10.2%}")
    lines.append(f"Max DD Duration:  {metrics.max_drawdown_duration:>10d} days")
    lines.append(f"Volatility:       {metrics.volatility:>10.2%}")
    lines.append(f"Win Rate:         {metrics.win_rate:>10.1%}")
    lines.append(f"Profit Factor:    {metrics.profit_factor:>10.2f}")
    lines.append(f"Trades:           {metrics.num_trades:>10d}")

    output_text = "\n".join(lines)
    print(output_text)

    # Save output
    output_path = OUTPUT_DIR / "output.txt"
    output_path.write_text(output_text)
    print(f"\nSaved to {output_path}")

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity, linewidth=1.2)
    ax.set_title("Equity Curve (AAPL, 1yr backtest)")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value ($)")
    ax.axhline(
        y=config.initial_capital,
        color="gray",
        linestyle="--",
        linewidth=0.8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    chart_path = OUTPUT_DIR / "equity_curve.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {chart_path}")


if __name__ == "__main__":
    asyncio.run(main())
