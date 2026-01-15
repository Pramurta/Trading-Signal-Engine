# Algorithmic Trading Signal Engine

A production-ready Python framework for real-time market data processing, statistical signal generation, and risk-managed trading strategy development.

## Overview

This project demonstrates core quantitative development skills:
- **Asynchronous data processing** for concurrent market data streams
- **Statistical signal generation** using proven technical indicators
- **Risk management** with position sizing and drawdown controls
- **Vectorized backtesting** for fast historical strategy evaluation

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Handler  │────▶│  Signal Engine  │────▶│  Risk Manager   │
│   (async I/O)   │     │  (statistics)   │     │  (sizing/limits)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │    Strategy     │
                        │  (orchestrator) │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Backtester    │
                        │  (vectorized)   │
                        └─────────────────┘
```

## Key Features

### 1. Asynchronous Market Data Processing
Non-blocking concurrent data fetching using `asyncio` for handling multiple symbols simultaneously without I/O bottlenecks.

### 2. Statistical Signal Generators
- **Z-Score**: Mean reversion signal measuring standard deviations from rolling mean
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **Bollinger Bands**: Volatility-adjusted price channels
- **MACD**: Trend-following momentum indicator

### 3. Risk Management
- **Kelly Criterion**: Optimal position sizing based on win rate and payoff ratio
- **Volatility Scaling**: Position sizes adjusted for current market volatility
- **Maximum Drawdown Limits**: Automatic risk reduction during drawdowns

### 4. Vectorized Backtesting
NumPy-based backtesting engine for fast historical simulation with realistic transaction costs.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading_signal_engine.git
cd trading_signal_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from src.data_handler import MarketDataHandler
from src.signals import SignalEngine
from src.risk_manager import RiskManager
from src.strategy import TradingStrategy

# Initialize components
data_handler = MarketDataHandler()
signal_engine = SignalEngine()
risk_manager = RiskManager(max_position_pct=0.1, max_drawdown_pct=0.15)

# Create strategy
strategy = TradingStrategy(
    data_handler=data_handler,
    signal_engine=signal_engine,
    risk_manager=risk_manager
)

# Run backtest
async def main():
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    results = await strategy.run_backtest(symbols, days=252)
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Return: {results['total_return']:.2%}")

asyncio.run(main())
```

## Project Structure

```
trading_signal_engine/
├── src/
│   ├── __init__.py
│   ├── data_handler.py      # Async market data processing
│   ├── signals.py           # Statistical signal generators
│   ├── risk_manager.py      # Position sizing & risk limits
│   ├── strategy.py          # Trading strategy orchestration
│   └── backtest.py          # Vectorized backtesting engine
├── tests/
│   ├── __init__.py
│   └── test_signals.py      # Unit tests for signal generators
├── examples/
│   └── run_backtest.py      # Demo script with visualization
├── README.md
└── requirements.txt
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src
```

## Example Output

```
================================================================================
                    TRADING SIGNAL ENGINE - BACKTEST RESULTS
================================================================================

Performance Metrics:
  Total Return:      23.45%
  Sharpe Ratio:      1.87
  Max Drawdown:      -8.32%
  Win Rate:          54.2%
  Profit Factor:     1.65

Risk Metrics:
  Volatility (Ann):  12.4%
  VaR (95%):         -1.82%
  Avg Position Size: 8.5%
```

## Technical Concepts

### Z-Score Signal
The Z-score measures how many standard deviations the current price is from its rolling mean:

```
z = (price - rolling_mean) / rolling_std
```

Trading logic:
- Z < -2: Strong buy signal (price unusually low)
- Z > +2: Strong sell signal (price unusually high)

### Kelly Criterion
Optimal fraction of capital to risk:

```
f* = (p * b - q) / b

where:
  p = probability of winning
  q = probability of losing (1 - p)
  b = win/loss ratio
```

### Volatility-Adjusted Position Sizing
Position size inversely proportional to recent volatility:

```
position_size = target_risk / (volatility * price)
```

## Performance Considerations

- **Vectorized operations**: All signal calculations use NumPy for speed
- **Async I/O**: Non-blocking data fetching for multiple symbols
- **Memory efficiency**: Rolling calculations avoid storing full history

## Future Enhancements

- [ ] WebSocket integration for live market data
- [ ] Multi-asset correlation analysis
- [ ] Machine learning signal integration
- [ ] Order execution simulation with slippage models

## License

MIT License - feel free to use and modify for your own projects.

## Author

Built as a demonstration of quantitative development skills for algorithmic trading roles.
