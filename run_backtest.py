#!/usr/bin/env python3
"""Demo script — runs all components of the trading signal engine end-to-end."""

import asyncio

import numpy as np

from src.backtest import BacktestConfig, Backtester
from src.data_handler import MarketDataHandler
from src.risk_manager import Portfolio, RiskManager
from src.signals import SignalEngine


def print_header(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


async def demo_data_handler():
    print_header("Data Handler (synthetic)")

    handler = MarketDataHandler(random_seed=42)
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    print(f"\nFetching {len(symbols)} symbols concurrently...")
    data = await handler.stream_market_data(symbols, days=252)

    for symbol, market_data in data.items():
        returns = np.diff(np.log(market_data.close))
        annual_vol = np.std(returns) * np.sqrt(252)
        total_return = (market_data.close[-1] / market_data.close[0]) - 1
        print(
            f"  {symbol:5s}: {len(market_data):3d} days | "
            f"Vol: {annual_vol:5.1%} | Return: {total_return:+6.1%}"
        )

    return data


def demo_signals(data: dict):
    print_header("Signal Generation")

    engine = SignalEngine()
    symbol = list(data.keys())[0]
    prices = data[symbol].close

    zscore = engine.calculate_zscore(prices, window=20)
    rsi = engine.calculate_rsi(prices, window=14)
    macd = engine.calculate_macd(prices)
    bollinger, _, _, _ = engine.calculate_bollinger_bands(prices)

    combined = engine.combine_signals(
        {"zscore": zscore, "rsi": rsi, "macd": macd, "bollinger": bollinger},
        weights={"zscore": 0.35, "rsi": 0.25, "macd": 0.25, "bollinger": 0.15},
    )

    print(f"\nSignals for {symbol}:")
    for name, sig in [
        ("Z-Score", zscore),
        ("RSI", rsi),
        ("MACD", macd),
        ("Bollinger", bollinger),
        ("Combined", combined),
    ]:
        val = sig.get_current()
        print(f"  {name:10s}  {val:+.3f}  (str: {sig.strength:.2f})")

    direction_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
    print(f"  Direction:  {direction_map[combined.get_direction()]}")


def demo_risk_management(data: dict):
    print_header("Risk Management")

    risk_mgr = RiskManager(
        max_position_pct=0.10, max_drawdown_pct=0.15, kelly_fraction=0.25
    )
    portfolio = Portfolio(capital=100_000)

    symbol = list(data.keys())[0]
    prices = data[symbol].close
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(252)

    position = risk_mgr.calculate_position_size(
        symbol=symbol,
        signal_strength=0.75,
        price=prices[-1],
        volatility=volatility,
        portfolio=portfolio,
        win_rate=0.55,
        win_loss_ratio=1.5,
    )

    print(f"\nPortfolio: ${portfolio.capital:,.0f}")
    print(f"Asset: {symbol} @ ${prices[-1]:.2f} (vol: {volatility:.1%})")
    print("\nPosition sizing (signal=0.75, win_rate=55%, W/L=1.5x):")
    print(f"  Shares: {position.shares:,}")
    print(f"  Notional: ${position.notional:,.0f}")
    print(f"  Weight: {position.weight:.1%}")
    print(f"  Risk Score: {position.risk_score:.2f}")

    kelly_full = risk_mgr._kelly_criterion(0.55, 1.5)
    print(f"\nKelly: full={kelly_full:.1%}, quarter={kelly_full * 0.25:.1%}")


def demo_backtest(data: dict):
    print_header("Vectorized Backtest")

    engine = SignalEngine()
    symbols = list(data.keys())[:4]
    min_length = min(len(data[s].close) for s in symbols)

    # Build T x N matrices
    prices = np.column_stack([data[s].close[:min_length] for s in symbols])
    signals = np.column_stack(
        [
            engine.combine_signals(
                {
                    "zscore": engine.calculate_zscore(data[s].close[:min_length]),
                    "rsi": engine.calculate_rsi(data[s].close[:min_length]),
                    "macd": engine.calculate_macd(data[s].close[:min_length]),
                    "bollinger": engine.calculate_bollinger_bands(
                        data[s].close[:min_length]
                    )[0],
                },
                weights={"zscore": 0.35, "rsi": 0.25, "macd": 0.25, "bollinger": 0.15},
            ).values
            for s in symbols
        ]
    )

    config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=10,
        max_position_weight=0.25,
    )

    backtester = Backtester(config)
    metrics, equity_curve, _ = backtester.run(prices, signals, asset_names=symbols)

    n = len(symbols)
    cap = config.initial_capital
    print(f"\n{n} assets, {min_length} days, ${cap:,.0f} initial")
    print(f"Final equity: ${equity_curve[-1]:,.0f}")
    print(f"\n  Total Return:     {metrics.total_return:>10.2%}")
    print(f"  CAGR:             {metrics.cagr:>10.2%}")
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:    {metrics.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:     {metrics.max_drawdown:>10.2%}")
    print(f"  Max DD Duration:  {metrics.max_drawdown_duration:>10d} days")
    print(f"  Volatility:       {metrics.volatility:>10.2%}")
    print(f"  Win Rate:         {metrics.win_rate:>10.1%}")
    print(f"  Profit Factor:    {metrics.profit_factor:>10.2f}")
    print(f"  Trades:           {metrics.num_trades:>10d}")


async def main():
    data = await demo_data_handler()
    demo_signals(data)
    demo_risk_management(data)
    demo_backtest(data)
    print()


if __name__ == "__main__":
    asyncio.run(main())
