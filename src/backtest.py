"""
Vectorized Backtesting Engine

This module provides a standalone backtesting framework optimized for
speed using NumPy vectorized operations. It's designed for rapid
strategy iteration and parameter optimization.

Key Features:
    - Fully vectorized calculations (no loops over time)
    - Support for multiple assets
    - Realistic transaction cost modeling
    - Comprehensive performance analytics
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestConfig:
    """
    Configuration for backtest execution.

    Attributes:
        initial_capital: Starting portfolio value
        transaction_cost_bps: Round-trip transaction cost in basis points
        slippage_bps: Estimated slippage in basis points
        max_position_weight: Maximum weight per position
        risk_free_rate: Annual risk-free rate
    """

    initial_capital: float = 100_000
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.20
    risk_free_rate: float = 0.05


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    volatility: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    num_trades: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "num_trades": self.num_trades,
        }

    def __str__(self) -> str:
        return (
            f"Total Return: {self.total_return:.2%} | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Max DD: {self.max_drawdown:.2%}"
        )


class Backtester:
    """
    Vectorized backtesting engine for trading strategies.

    This class implements a high-performance backtesting framework
    using NumPy vectorized operations. It avoids Python loops over
    time periods for maximum speed.

    Features:
        - Signal-based position sizing
        - Transaction cost modeling
        - Comprehensive analytics
        - Walk-forward analysis support

    Example:
        >>> backtester = Backtester(config=BacktestConfig())
        >>> # prices: T x N array (T timepoints, N assets)
        >>> # signals: T x N array of signals in [-1, 1]
        >>> metrics, equity, positions = backtester.run(prices, signals)
        >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    """

    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize backtester with configuration.

        Args:
            config: BacktestConfig instance (uses defaults if None)
        """
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        asset_names: list[str] | None = None,
    ) -> tuple[BacktestMetrics, np.ndarray, np.ndarray]:
        """
        Execute vectorized backtest.

        This is the main entry point for running a backtest.
        It calculates positions from signals, computes returns
        with transaction costs, and generates performance metrics.

        Args:
            prices: Price matrix (T x N) for T periods, N assets
            signals: Signal matrix (T x N) with values in [-1, 1]
            asset_names: Optional list of asset names

        Returns:
            Tuple of (metrics, equity_curve, positions)

        Algorithm:
            1. Convert signals to target positions (with sizing)
            2. Calculate position changes for transaction costs
            3. Compute asset returns
            4. Calculate portfolio returns including costs
            5. Generate equity curve
            6. Compute all performance metrics
        """
        # Handle 1D arrays (single asset)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            signals = signals.reshape(-1, 1)

        T, N = prices.shape

        if signals.shape != prices.shape:
            raise ValueError(
                f"Signal shape {signals.shape} != price shape {prices.shape}"
            )

        # Step 1: Convert signals to positions
        signals = np.clip(signals, -1, 1)

        # Equal weight across assets, scaled by signal
        base_weight = self.config.max_position_weight / N
        positions = signals * base_weight  # T x N position weights

        # Step 2: Calculate asset returns
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        # Step 3: Calculate position changes for transaction costs
        position_changes = np.zeros_like(positions)
        position_changes[1:] = np.abs(positions[1:] - positions[:-1])

        # Total turnover per period
        turnover = np.sum(position_changes, axis=1)

        # Transaction costs
        total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        transaction_costs = turnover * (total_cost_bps / 10000)

        # Step 4: Calculate portfolio returns
        lagged_positions = np.zeros_like(positions)
        lagged_positions[1:] = positions[:-1]

        portfolio_returns = (
            np.sum(lagged_positions * returns, axis=1) - transaction_costs
        )

        # Step 5: Generate equity curve
        equity_multipliers = 1 + portfolio_returns
        equity_curve = self.config.initial_capital * np.cumprod(equity_multipliers)

        # Step 6: Calculate metrics
        metrics = self._calculate_metrics(portfolio_returns, equity_curve, positions)

        return metrics, equity_curve, positions

    def _calculate_metrics(
        self, returns: np.ndarray, equity_curve: np.ndarray, positions: np.ndarray
    ) -> BacktestMetrics:
        """
        Calculate comprehensive performance metrics.

        Metrics calculated:
            - Total return and CAGR
            - Sharpe and Sortino ratios
            - Maximum drawdown and duration
            - Calmar ratio
            - Win rate and profit factor
        """
        T = len(returns)

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # CAGR (assumes 252 trading days per year)
        years = T / 252
        if years > 0 and equity_curve[0] > 0:
            cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1
        else:
            cagr = 0

        # Volatility
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        # Sharpe ratio
        daily_rf = self.config.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = (
            (np.mean(excess_returns) / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns) * np.sqrt(252)
            sortino = (
                (np.mean(excess_returns) * 252 / downside_vol)
                if downside_vol > 0
                else 0
            )
        else:
            sortino = 0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)

        # Maximum drawdown duration
        underwater = drawdown > 0
        max_dd_duration = self._max_consecutive_true(underwater)

        # Calmar ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        trade_returns = self._calculate_trade_returns(returns, positions)

        num_trades = len(trade_returns)
        win_rate = np.mean(trade_returns > 0) if num_trades > 0 else 0

        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]

        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        gross_loss = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade_return = np.mean(trade_returns) if num_trades > 0 else 0

        return BacktestMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            volatility=annual_vol,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            num_trades=num_trades,
        )

    def _calculate_trade_returns(
        self, returns: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Extract individual trade returns."""
        total_position = np.sum(positions, axis=1) if positions.ndim > 1 else positions
        position_direction = np.sign(total_position)

        direction_changes = np.diff(position_direction, prepend=0) != 0
        trade_ids = np.cumsum(direction_changes)

        trade_returns = []
        for trade_id in np.unique(trade_ids):
            if trade_id == 0:
                continue

            mask = trade_ids == trade_id
            if np.sum(mask) > 0 and np.sum(total_position[mask]) != 0:
                trade_return = np.sum(returns[mask])
                trade_returns.append(trade_return)

        return np.array(trade_returns) if trade_returns else np.array([0.0])

    @staticmethod
    def _max_consecutive_true(arr: np.ndarray) -> int:
        """Find maximum consecutive True values in boolean array."""
        if len(arr) == 0:
            return 0

        arr_int = arr.astype(int)
        padded = np.concatenate([[0], arr_int, [0]])

        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return 0

        run_lengths = ends - starts
        return int(np.max(run_lengths))

    def walk_forward_analysis(
        self,
        prices: np.ndarray,
        signal_generator: Callable[[np.ndarray], np.ndarray],
        train_periods: int = 126,
        test_periods: int = 21,
        step: int = 21,
    ) -> list[BacktestMetrics]:
        """
        Perform walk-forward analysis for out-of-sample testing.

        Args:
            prices: Price matrix (T x N)
            signal_generator: Function that generates signals from prices
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step: Number of periods to step forward

        Returns:
            List of BacktestMetrics for each test window
        """
        T = len(prices)
        results = []

        start = train_periods
        while start + test_periods <= T:
            full_signals = signal_generator(prices[: start + test_periods])

            test_prices = prices[start : start + test_periods]
            test_signals = full_signals[start : start + test_periods]

            metrics, _, _ = self.run(test_prices, test_signals)
            results.append(metrics)

            start += step

        return results

    def parameter_sweep(
        self,
        prices: np.ndarray,
        signal_generator: Callable[[np.ndarray, dict], np.ndarray],
        param_grid: dict[str, list],
        metric: str = "sharpe_ratio",
    ) -> tuple[dict, BacktestMetrics]:
        """
        Sweep over parameter combinations to find optimal settings.

        Args:
            prices: Price matrix
            signal_generator: Function(prices, params) -> signals
            param_grid: Dictionary of parameter names to value lists
            metric: Metric to optimize

        Returns:
            Tuple of (best_params, best_metrics)
        """
        best_params = None
        best_metrics = None
        best_value = float("-inf")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            signals = signal_generator(prices, params)
            metrics, _, _ = self.run(prices, signals)

            metric_value = getattr(metrics, metric)
            if metric_value > best_value:
                best_value = metric_value
                best_params = params
                best_metrics = metrics

        return best_params, best_metrics


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Vectorized Backtester - Demo")
    print("=" * 60)

    # Generate synthetic multi-asset data
    np.random.seed(42)
    T = 252  # 1 year
    N = 4  # 4 assets

    # Generate correlated returns
    mean_returns = np.array([0.0003, 0.0002, 0.0004, 0.0001])
    volatilities = np.array([0.02, 0.015, 0.025, 0.018])

    # Correlation matrix
    corr = np.array(
        [
            [1.0, 0.6, 0.4, 0.3],
            [0.6, 1.0, 0.5, 0.4],
            [0.4, 0.5, 1.0, 0.6],
            [0.3, 0.4, 0.6, 1.0],
        ]
    )

    # Cholesky decomposition for correlated samples
    L = np.linalg.cholesky(corr)

    # Generate returns
    uncorrelated = np.random.randn(T, N)
    correlated = uncorrelated @ L.T
    returns = mean_returns + volatilities * correlated

    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    # Generate simple momentum signals
    def momentum_signal(prices: np.ndarray, lookback: int = 20) -> np.ndarray:
        """Simple momentum signal based on recent returns."""
        signals = np.zeros_like(prices)
        for i in range(lookback, len(prices)):
            past_returns = (prices[i] - prices[i - lookback]) / prices[i - lookback]
            signals[i] = np.tanh(past_returns * 10)  # Scale and bound
        return signals

    signals = momentum_signal(prices)

    # Run backtest
    config = BacktestConfig(
        initial_capital=100_000, transaction_cost_bps=10, max_position_weight=0.25
    )

    backtester = Backtester(config)
    metrics, equity_curve, positions = backtester.run(prices, signals)

    # Print results
    print(f"\nBacktest Period: {T} days ({T / 252:.1f} years)")
    print(f"Number of Assets: {N}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print(f"Final Capital: ${equity_curve[-1]:,.0f}")

    print("\n" + "-" * 40)
    print("Performance Metrics")
    print("-" * 40)
    print(f"Total Return:       {metrics.total_return:>10.2%}")
    print(f"CAGR:               {metrics.cagr:>10.2%}")
    print(f"Sharpe Ratio:       {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:      {metrics.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:       {metrics.max_drawdown:>10.2%}")
    print(f"Max DD Duration:    {metrics.max_drawdown_duration:>10d} days")
    print(f"Calmar Ratio:       {metrics.calmar_ratio:>10.2f}")
    print(f"Volatility:         {metrics.volatility:>10.2%}")
    print(f"Win Rate:           {metrics.win_rate:>10.1%}")
    print(f"Profit Factor:      {metrics.profit_factor:>10.2f}")
    print(f"Avg Trade Return:   {metrics.avg_trade_return:>10.4f}")
    print(f"Number of Trades:   {metrics.num_trades:>10d}")

    print("\n" + "=" * 60)
