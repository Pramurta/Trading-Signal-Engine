"""
Trading Strategy Orchestrator

This module ties together data handling, signal generation, and risk
management into a cohesive trading strategy framework.

Key Responsibilities:
    - Coordinate async data fetching
    - Generate and combine trading signals
    - Apply risk management rules
    - Execute backtests
    - Generate trade decisions
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_handler import MarketDataHandler, MarketData
from .signals import SignalEngine, Signal
from .risk_manager import RiskManager, Portfolio, PositionSize, RiskMetrics


@dataclass
class TradeDecision:
    """
    A complete trade decision with full context.
    
    Attributes:
        symbol: Trading symbol
        direction: 1 (long), -1 (short), 0 (no trade)
        position_size: Recommended position
        signals: Individual signal values
        combined_signal: Weighted combination
        timestamp: When decision was made
        reason: Human-readable explanation
    """
    symbol: str
    direction: int
    position_size: PositionSize
    signals: Dict[str, float]
    combined_signal: float
    timestamp: datetime
    reason: str
    
    def __str__(self) -> str:
        direction_str = {1: "LONG", -1: "SHORT", 0: "HOLD"}[self.direction]
        return (
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{self.symbol} | {direction_str} | "
            f"Signal: {self.combined_signal:+.2f} | "
            f"{self.position_size}"
        )


@dataclass
class BacktestResult:
    """
    Complete backtest results with performance metrics.
    
    Attributes:
        total_return: Total percentage return
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum peak-to-trough decline
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / gross loss
        num_trades: Total number of trades
        equity_curve: Daily portfolio values
        returns: Daily return series
    """
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    equity_curve: np.ndarray
    returns: np.ndarray
    risk_metrics: RiskMetrics
    trade_log: List[TradeDecision]


class TradingStrategy:
    """
    Main trading strategy orchestrator.
    
    This class coordinates all components of the trading system:
        1. Fetches market data asynchronously
        2. Calculates technical signals
        3. Combines signals with custom weights
        4. Applies risk management
        5. Generates trade decisions
        6. Runs backtests
    
    Example:
        >>> data_handler = MarketDataHandler()
        >>> signal_engine = SignalEngine()
        >>> risk_manager = RiskManager()
        >>> 
        >>> strategy = TradingStrategy(
        ...     data_handler=data_handler,
        ...     signal_engine=signal_engine,
        ...     risk_manager=risk_manager
        ... )
        >>> 
        >>> results = await strategy.run_backtest(['AAPL', 'GOOGL'], days=252)
    """
    
    def __init__(
        self,
        data_handler: MarketDataHandler,
        signal_engine: SignalEngine,
        risk_manager: RiskManager,
        signal_weights: Optional[Dict[str, float]] = None,
        signal_threshold: float = 0.3,
        transaction_cost_bps: float = 5.0
    ):
        """
        Initialize the trading strategy.
        
        Args:
            data_handler: Async market data handler
            signal_engine: Signal calculation engine
            risk_manager: Risk management system
            signal_weights: Custom weights for signal combination
            signal_threshold: Minimum signal strength to trade
            transaction_cost_bps: Transaction costs in basis points
        """
        self.data_handler = data_handler
        self.signal_engine = signal_engine
        self.risk_manager = risk_manager
        
        # Default equal weights
        self.signal_weights = signal_weights or {
            'zscore': 0.35,
            'rsi': 0.25,
            'macd': 0.25,
            'bollinger': 0.15
        }
        
        self.signal_threshold = signal_threshold
        self.transaction_cost_bps = transaction_cost_bps
    
    async def generate_signals(
        self,
        symbol: str,
        data: MarketData
    ) -> Tuple[Dict[str, Signal], Signal]:
        """
        Generate all signals for a symbol.
        
        This calculates multiple technical indicators and combines
        them into a single trading signal.
        
        Args:
            symbol: Trading symbol
            data: Market data for the symbol
        
        Returns:
            Tuple of (individual signals dict, combined signal)
        """
        prices = data.close
        
        # Calculate individual signals
        zscore = self.signal_engine.calculate_zscore(prices, window=20)
        rsi = self.signal_engine.calculate_rsi(prices, window=14)
        macd = self.signal_engine.calculate_macd(prices)
        bollinger, _, _, _ = self.signal_engine.calculate_bollinger_bands(prices)
        
        signals = {
            'zscore': zscore,
            'rsi': rsi,
            'macd': macd,
            'bollinger': bollinger
        }
        
        # Combine signals
        combined = self.signal_engine.combine_signals(signals, self.signal_weights)
        
        return signals, combined
    
    async def generate_trade_decision(
        self,
        symbol: str,
        data: MarketData,
        portfolio: Portfolio,
        win_rate: float = 0.52,
        win_loss_ratio: float = 1.3
    ) -> TradeDecision:
        """
        Generate a complete trade decision for a symbol.
        
        This is the main decision-making function that:
            1. Calculates signals
            2. Determines trade direction
            3. Calculates position size
            4. Applies risk checks
        
        Args:
            symbol: Trading symbol
            data: Market data
            portfolio: Current portfolio state
            win_rate: Historical win rate for Kelly calculation
            win_loss_ratio: Historical win/loss ratio
        
        Returns:
            Complete TradeDecision object
        """
        # Generate signals
        signals, combined = await self.generate_signals(symbol, data)
        
        current_price = data.close[-1]
        current_signal = combined.get_current()
        
        # Calculate asset volatility
        returns = np.diff(np.log(data.close))
        volatility = np.std(returns) * np.sqrt(252)
        
        # Determine direction
        if abs(current_signal) < self.signal_threshold:
            direction = 0
            reason = f"Signal {current_signal:.2f} below threshold {self.signal_threshold}"
        elif current_signal > 0:
            direction = 1
            reason = f"Bullish signal: {current_signal:.2f}"
        else:
            direction = -1
            reason = f"Bearish signal: {current_signal:.2f}"
        
        # Check if we should reduce risk
        if self.risk_manager.should_reduce_risk(portfolio):
            direction = 0
            reason = f"Risk reduction mode (drawdown: {portfolio.current_drawdown:.1%})"
        
        # Calculate position size
        position = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=abs(current_signal),
            price=current_price,
            volatility=volatility,
            portfolio=portfolio,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio
        )
        
        # Extract signal values for logging
        signal_values = {
            name: signal.get_current() 
            for name, signal in signals.items()
        }
        
        return TradeDecision(
            symbol=symbol,
            direction=direction,
            position_size=position,
            signals=signal_values,
            combined_signal=current_signal,
            timestamp=datetime.now(),
            reason=reason
        )
    
    async def run_backtest(
        self,
        symbols: List[str],
        days: int = 252,
        initial_capital: float = 100_000,
        rebalance_frequency: int = 1  # Daily
    ) -> BacktestResult:
        """
        Run a vectorized backtest on historical data.
        
        This simulates the strategy over historical data:
            1. Fetches historical data for all symbols
            2. Calculates signals at each point in time
            3. Simulates trades with transaction costs
            4. Tracks portfolio performance
        
        Args:
            symbols: List of symbols to trade
            days: Number of historical days
            initial_capital: Starting portfolio value
            rebalance_frequency: Days between rebalancing
        
        Returns:
            BacktestResult with full performance analysis
        """
        # Fetch all data concurrently
        market_data = await self.data_handler.stream_market_data(symbols, days)
        
        if not market_data:
            raise ValueError("No market data available for backtest")
        
        # Initialize tracking
        portfolio = Portfolio(capital=initial_capital)
        equity_curve = [initial_capital]
        daily_returns = []
        trade_log = []
        
        # Get common length across all symbols
        min_length = min(len(data.close) for data in market_data.values())
        
        # Track positions and PnL
        positions: Dict[str, int] = {s: 0 for s in symbols}
        entry_prices: Dict[str, float] = {s: 0.0 for s in symbols}
        
        # Win/loss tracking
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        # Simulate each day
        for day in range(50, min_length):  # Start at day 50 for indicator warmup
            daily_pnl = 0.0
            
            # Process each symbol
            for symbol, data in market_data.items():
                current_price = data.close[day]
                
                # Calculate signal using data up to current day
                truncated_data = MarketData(
                    symbol=symbol,
                    timestamp=data.timestamp[:day+1],
                    open=data.open[:day+1],
                    high=data.high[:day+1],
                    low=data.low[:day+1],
                    close=data.close[:day+1],
                    volume=data.volume[:day+1]
                )
                
                _, combined = await self.generate_signals(symbol, truncated_data)
                current_signal = combined.get_current()
                
                # Get current position
                current_pos = positions[symbol]
                
                # Determine target position based on signal
                if day % rebalance_frequency == 0:
                    if abs(current_signal) >= self.signal_threshold:
                        target_direction = 1 if current_signal > 0 else -1
                    else:
                        target_direction = 0
                    
                    # Simple position: fully in or fully out
                    max_shares = int(
                        (portfolio.total_equity * self.risk_manager.max_position_pct) 
                        / current_price / len(symbols)
                    )
                    
                    target_pos = target_direction * max_shares
                    
                    # Execute trade if position changes
                    if target_pos != current_pos:
                        # Calculate trade
                        trade_shares = target_pos - current_pos
                        trade_value = abs(trade_shares) * current_price
                        
                        # Transaction cost
                        cost = trade_value * (self.transaction_cost_bps / 10000)
                        daily_pnl -= cost
                        
                        # Track entry for PnL calculation
                        if target_pos != 0 and current_pos == 0:
                            entry_prices[symbol] = current_price
                        elif target_pos == 0 and current_pos != 0:
                            # Closing position - calculate PnL
                            trade_pnl = (current_price - entry_prices[symbol]) * current_pos
                            if trade_pnl > 0:
                                wins += 1
                                gross_profit += trade_pnl
                            else:
                                losses += 1
                                gross_loss += abs(trade_pnl)
                        
                        positions[symbol] = target_pos
                
                # Mark-to-market PnL
                if positions[symbol] != 0:
                    prev_price = data.close[day - 1]
                    daily_pnl += positions[symbol] * (current_price - prev_price)
            
            # Update portfolio
            new_equity = portfolio.total_equity + daily_pnl
            if new_equity > portfolio.peak_equity:
                portfolio.peak_equity = new_equity
            portfolio.capital = new_equity
            
            equity_curve.append(new_equity)
            
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
        
        # Convert to numpy arrays
        equity_array = np.array(equity_curve)
        returns_array = np.array(daily_returns)
        
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            returns_array, equity_array
        )
        
        # Calculate performance metrics
        total_return = (equity_array[-1] - initial_capital) / initial_capital
        
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=risk_metrics.sharpe_ratio,
            max_drawdown=risk_metrics.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=total_trades,
            equity_curve=equity_array,
            returns=returns_array,
            risk_metrics=risk_metrics,
            trade_log=trade_log
        )
    
    def print_backtest_results(self, results: BacktestResult) -> str:
        """
        Generate a formatted report of backtest results.
        
        Args:
            results: BacktestResult from run_backtest
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 70)
        report.append("                 TRADING SIGNAL ENGINE - BACKTEST RESULTS")
        report.append("=" * 70)
        report.append("")
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"  Total Return:      {results.total_return:>10.2%}")
        report.append(f"  Sharpe Ratio:      {results.sharpe_ratio:>10.2f}")
        report.append(f"  Max Drawdown:      {results.max_drawdown:>10.2%}")
        report.append(f"  Win Rate:          {results.win_rate:>10.1%}")
        report.append(f"  Profit Factor:     {results.profit_factor:>10.2f}")
        report.append(f"  Number of Trades:  {results.num_trades:>10d}")
        report.append("")
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"  Volatility (Ann):  {results.risk_metrics.volatility:>10.1%}")
        report.append(f"  VaR (95%):         {results.risk_metrics.var_95:>10.2%}")
        report.append(f"  Current Drawdown:  {results.risk_metrics.current_drawdown:>10.2%}")
        report.append(f"  Risk Level:        {results.risk_metrics.risk_level.value:>10s}")
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


# Example usage
async def example_strategy():
    """Demonstrate strategy usage."""
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
        risk_manager=risk_manager
    )
    
    # Run backtest
    print("Running backtest...")
    results = await strategy.run_backtest(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        days=252,
        initial_capital=100_000
    )
    
    # Print results
    print(strategy.print_backtest_results(results))
    
    return results


if __name__ == "__main__":
    asyncio.run(example_strategy())
