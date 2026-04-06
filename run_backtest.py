#!/usr/bin/env python3
"""
Trading Signal Engine - Complete Demo

This script demonstrates all components of the trading signal engine:
    1. Async data fetching
    2. Signal generation
    3. Risk management
    4. Backtesting
    5. Results visualization

Run with: python run_backtest.py
"""

import asyncio

import numpy as np

from src.data_handler import MarketDataHandler
from src.signals import SignalEngine
from src.risk_manager import RiskManager, Portfolio
from src.strategy import TradingStrategy
from src.backtest import Backtester, BacktestConfig


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 70
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{text}")
    print("-" * 40)


async def demo_data_handler():
    """Demonstrate async data fetching."""
    print_header("1. ASYNC DATA HANDLER DEMO")
    
    handler = MarketDataHandler(random_seed=42)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    print(f"\nFetching data for {len(symbols)} symbols concurrently...")
    data = await handler.stream_market_data(symbols, days=252)
    
    print_section("Data Summary")
    for symbol, market_data in data.items():
        returns = np.diff(np.log(market_data.close))
        annual_vol = np.std(returns) * np.sqrt(252)
        total_return = (market_data.close[-1] / market_data.close[0]) - 1
        
        print(f"  {symbol:5s}: {len(market_data):3d} days | "
              f"Vol: {annual_vol:5.1%} | Return: {total_return:+6.1%}")
    
    return data


def demo_signals(data: dict):
    """Demonstrate signal generation."""
    print_header("2. SIGNAL GENERATION DEMO")
    
    engine = SignalEngine()
    
    # Use first symbol for demonstration
    symbol = list(data.keys())[0]
    prices = data[symbol].close
    
    print(f"\nGenerating signals for {symbol}...")
    
    # Calculate individual signals
    zscore = engine.calculate_zscore(prices, window=20)
    rsi = engine.calculate_rsi(prices, window=14)
    macd = engine.calculate_macd(prices)
    bollinger, upper, middle, lower = engine.calculate_bollinger_bands(prices)
    
    # Combine signals
    combined = engine.combine_signals({
        'zscore': zscore,
        'rsi': rsi,
        'macd': macd,
        'bollinger': bollinger
    }, weights={'zscore': 0.35, 'rsi': 0.25, 'macd': 0.25, 'bollinger': 0.15})
    
    print_section("Current Signal Values")
    print(f"  Z-Score:    {zscore.get_current():+.3f}  (strength: {zscore.strength:.2f})")
    print(f"  RSI:        {rsi.get_current():+.3f}  (strength: {rsi.strength:.2f})")
    print(f"  MACD:       {macd.get_current():+.3f}  (strength: {macd.strength:.2f})")
    print(f"  Bollinger:  {bollinger.get_current():+.3f}  (strength: {bollinger.strength:.2f})")
    print(f"  ─────────────────────────────")
    print(f"  Combined:   {combined.get_current():+.3f}  (strength: {combined.strength:.2f})")
    
    direction_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
    print(f"\n  Trade Direction: {direction_map[combined.get_direction()]}")
    
    return engine


def demo_risk_management(data: dict):
    """Demonstrate risk management."""
    print_header("3. RISK MANAGEMENT DEMO")
    
    risk_mgr = RiskManager(
        max_position_pct=0.10,
        max_drawdown_pct=0.15,
        kelly_fraction=0.25
    )
    
    # Create portfolio
    portfolio = Portfolio(capital=100_000)
    
    # Get sample data
    symbol = list(data.keys())[0]
    prices = data[symbol].close
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(252)
    
    print(f"\nPortfolio: ${portfolio.capital:,.0f}")
    print(f"Asset: {symbol} @ ${prices[-1]:.2f}")
    print(f"Volatility: {volatility:.1%}")
    
    # Calculate position size with historical stats
    position = risk_mgr.calculate_position_size(
        symbol=symbol,
        signal_strength=0.75,
        price=prices[-1],
        volatility=volatility,
        portfolio=portfolio,
        win_rate=0.55,
        win_loss_ratio=1.5
    )
    
    print_section("Position Sizing")
    print(f"  Signal Strength: 0.75")
    print(f"  Win Rate: 55%")
    print(f"  Win/Loss Ratio: 1.5x")
    print(f"\n  Recommended Position:")
    print(f"    Shares: {position.shares:,}")
    print(f"    Notional: ${position.notional:,.0f}")
    print(f"    Portfolio Weight: {position.weight:.1%}")
    print(f"    Risk Score: {position.risk_score:.2f}")
    
    # Kelly Criterion breakdown
    kelly_full = risk_mgr._kelly_criterion(0.55, 1.5)
    print_section("Kelly Criterion Analysis")
    print(f"  Full Kelly: {kelly_full:.1%}")
    print(f"  Quarter Kelly: {kelly_full * 0.25:.1%}")
    print(f"  Max Position: {risk_mgr.max_position_pct:.1%}")
    
    # Simulate drawdown scenario
    print_section("Drawdown Scenario")
    dd_portfolio = Portfolio(capital=88_000)
    dd_portfolio.peak_equity = 100_000
    
    print(f"  Peak Equity: ${dd_portfolio.peak_equity:,.0f}")
    print(f"  Current Equity: ${dd_portfolio.capital:,.0f}")
    print(f"  Drawdown: {dd_portfolio.current_drawdown:.1%}")
    print(f"  Risk Budget: {risk_mgr.get_risk_budget(dd_portfolio):.1%}")
    print(f"  Reduce Risk: {risk_mgr.should_reduce_risk(dd_portfolio)}")
    
    return risk_mgr


async def demo_full_backtest():
    """Run complete strategy backtest."""
    print_header("4. FULL STRATEGY BACKTEST")
    
    # Initialize components
    data_handler = MarketDataHandler(random_seed=42)
    signal_engine = SignalEngine()
    risk_manager = RiskManager(
        max_position_pct=0.15,
        max_drawdown_pct=0.20
    )
    
    # Create strategy
    strategy = TradingStrategy(
        data_handler=data_handler,
        signal_engine=signal_engine,
        risk_manager=risk_manager,
        signal_threshold=0.3,
        transaction_cost_bps=5.0
    )
    
    # Run backtest
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    print(f"\nBacktesting {len(symbols)} symbols over 252 days...")
    print(f"Initial Capital: $100,000")
    print(f"Transaction Cost: 5 bps")
    
    results = await strategy.run_backtest(
        symbols=symbols,
        days=252,
        initial_capital=100_000
    )
    
    # Print results
    print(strategy.print_backtest_results(results))
    
    return results


def demo_vectorized_backtest():
    """Demonstrate standalone vectorized backtester."""
    print_header("5. VECTORIZED BACKTESTER DEMO")
    
    # Generate multi-asset data
    np.random.seed(42)
    T = 252  # 1 year
    N = 4    # 4 assets
    
    print(f"\nGenerating {N} correlated assets over {T} days...")
    
    # Generate correlated returns
    mean_returns = np.array([0.0003, 0.0002, 0.0004, 0.0001])
    volatilities = np.array([0.02, 0.015, 0.025, 0.018])
    
    # Correlation matrix
    corr = np.array([
        [1.0, 0.6, 0.4, 0.3],
        [0.6, 1.0, 0.5, 0.4],
        [0.4, 0.5, 1.0, 0.6],
        [0.3, 0.4, 0.6, 1.0]
    ])
    
    L = np.linalg.cholesky(corr)
    uncorrelated = np.random.randn(T, N)
    correlated = uncorrelated @ L.T
    returns = mean_returns + volatilities * correlated
    
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Generate momentum signals
    def momentum_signal(prices, lookback=20):
        signals = np.zeros_like(prices)
        for i in range(lookback, len(prices)):
            past_returns = (prices[i] - prices[i-lookback]) / prices[i-lookback]
            signals[i] = np.tanh(past_returns * 10)
        return signals
    
    signals = momentum_signal(prices)
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=10,
        max_position_weight=0.25
    )
    
    backtester = Backtester(config)
    metrics, equity_curve, positions = backtester.run(prices, signals)
    
    print_section("Vectorized Backtest Results")
    print(f"  Total Return:      {metrics.total_return:>10.2%}")
    print(f"  CAGR:              {metrics.cagr:>10.2%}")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown:>10.2%}")
    print(f"  Max DD Duration:   {metrics.max_drawdown_duration:>10d} days")
    print(f"  Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")
    print(f"  Volatility:        {metrics.volatility:>10.2%}")
    print(f"  Win Rate:          {metrics.win_rate:>10.1%}")
    print(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
    print(f"  Num Trades:        {metrics.num_trades:>10d}")
    
    # Parameter sweep demo
    print_section("Parameter Optimization (Demo)")
    
    def signal_with_params(prices, params):
        return momentum_signal(prices, lookback=params['lookback'])
    
    param_grid = {
        'lookback': [10, 15, 20, 25, 30]
    }
    
    print("  Sweeping lookback periods: [10, 15, 20, 25, 30]")
    best_params, best_metrics = backtester.parameter_sweep(
        prices, signal_with_params, param_grid
    )
    
    print(f"\n  Best Parameters: {best_params}")
    print(f"  Best Sharpe: {best_metrics.sharpe_ratio:.2f}")
    
    return metrics


def print_ascii_chart(data: np.ndarray, width: int = 60, height: int = 15):
    """Print a simple ASCII chart of the data."""
    # Normalize data to height
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
    
    # Sample data to fit width
    indices = np.linspace(0, len(data) - 1, width).astype(int)
    sampled = data[indices]
    
    # Create chart
    chart = []
    for row in range(height, 0, -1):
        threshold = min_val + (row / height) * range_val
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        chart.append(line)
    
    # Print chart
    print(f"  ${max_val:,.0f} ┤{chart[0]}")
    for i in range(1, height - 1):
        print(f"         │{chart[i]}")
    print(f"  ${min_val:,.0f} ┤{chart[-1]}")
    print(f"         └{'─' * width}")
    print(f"          {'Start':<{width//2}}{'End':>{width//2}}")


async def main():
    """Run all demonstrations."""
    print_header("TRADING SIGNAL ENGINE - COMPLETE DEMO", "═")
    print("\nThis demo showcases all components of the quantitative")
    print("trading system: data handling, signals, risk management,")
    print("and backtesting.")
    
    # Run demos
    data = await demo_data_handler()
    demo_signals(data)
    demo_risk_management(data)
    results = await demo_full_backtest()
    demo_vectorized_backtest()
    
    # Final summary
    print_header("DEMO COMPLETE", "═")
    print("\nKey Takeaways:")
    print("  • Async data handling enables concurrent API calls")
    print("  • Multiple signals combined for robust trade decisions")
    print("  • Kelly Criterion + volatility scaling for position sizing")
    print("  • Drawdown controls protect capital during losses")
    print("  • Vectorized backtesting enables rapid strategy iteration")
    
    print("\nEquity Curve (ASCII):")
    print_ascii_chart(results.equity_curve)
    
    print("\nNext Steps:")
    print("  1. Connect to real market data APIs")
    print("  2. Implement additional signal types")
    print("  3. Add ML-based signal combination")
    print("  4. Deploy with proper execution infrastructure")
    
    print("\n" + "═" * 70)


if __name__ == "__main__":
    asyncio.run(main())
