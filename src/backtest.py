"""Vectorized backtesting engine for price/signal matrices."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.20
    risk_free_rate: float = 0.05


@dataclass
class BacktestMetrics:
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
    """Vectorized backtester operating on (T x N) price/signal matrices."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        asset_names: list[str] | None = None,
    ) -> tuple[BacktestMetrics, np.ndarray, np.ndarray]:
        """Run backtest. Returns (metrics, equity_curve, positions)."""
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            signals = signals.reshape(-1, 1)

        T, N = prices.shape

        if signals.shape != prices.shape:
            raise ValueError(
                f"Signal shape {signals.shape} != price shape {prices.shape}"
            )

        signals = np.clip(signals, -1, 1)

        # Position weights: equal-weight base scaled by signal
        base_weight = self.config.max_position_weight / N
        positions = signals * base_weight

        # Asset returns
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        # Transaction costs from position changes
        position_changes = np.zeros_like(positions)
        position_changes[1:] = np.abs(positions[1:] - positions[:-1])
        turnover = np.sum(position_changes, axis=1)
        total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        transaction_costs = turnover * (total_cost_bps / 10000)

        # Portfolio returns using lagged positions (no lookahead)
        lagged_positions = np.zeros_like(positions)
        lagged_positions[1:] = positions[:-1]
        portfolio_returns = (
            np.sum(lagged_positions * returns, axis=1) - transaction_costs
        )

        # Equity curve
        equity_curve = self.config.initial_capital * np.cumprod(1 + portfolio_returns)

        metrics = self._calculate_metrics(portfolio_returns, equity_curve, positions)
        return metrics, equity_curve, positions

    def _calculate_metrics(
        self, returns: np.ndarray, equity_curve: np.ndarray, positions: np.ndarray
    ) -> BacktestMetrics:
        """Compute all performance metrics from returns and equity curve."""
        T = len(returns)

        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        years = T / 252
        if years > 0 and equity_curve[0] > 0:
            cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1
        else:
            cagr = 0

        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        daily_rf = self.config.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = (
            (np.mean(excess_returns) / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        )

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

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)

        underwater = drawdown > 0
        max_dd_duration = self._max_consecutive_true(underwater)

        calmar = cagr / max_drawdown if max_drawdown > 0 else 0

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
            win_rate=float(win_rate),
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            num_trades=num_trades,
        )

    def _calculate_trade_returns(
        self, returns: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Segment returns into trades by position direction changes."""
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
                trade_returns.append(np.sum(returns[mask]))

        return np.array(trade_returns) if trade_returns else np.array([0.0])

    @staticmethod
    def _max_consecutive_true(arr: np.ndarray) -> int:
        """Find the longest run of True values in a boolean array."""
        if len(arr) == 0:
            return 0
        arr_int = arr.astype(int)
        padded = np.concatenate([[0], arr_int, [0]])
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    def walk_forward_analysis(
        self,
        prices: np.ndarray,
        signal_generator: Callable[[np.ndarray], np.ndarray],
        train_periods: int = 126,
        test_periods: int = 21,
        step: int = 21,
    ) -> list[BacktestMetrics]:
        """Walk-forward out-of-sample test over rolling windows."""
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
    ) -> tuple[dict, BacktestMetrics | None]:
        """Brute-force parameter optimization. Returns (best_params, best_metrics)."""
        best_params: dict = {}
        best_metrics: BacktestMetrics | None = None
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
