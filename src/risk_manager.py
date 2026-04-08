"""Risk management — Kelly Criterion, vol scaling, drawdown controls."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionSize:
    """Recommended position with risk metrics."""

    symbol: str
    shares: int
    notional: float
    weight: float  # portfolio weight, 0 to 1
    risk_score: float  # 0 (low) to 1 (high)

    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.shares:,} shares "
            f"(${self.notional:,.0f}, {self.weight:.1%} weight)"
        )


@dataclass
class RiskMetrics:
    """Portfolio-level risk metrics."""

    current_drawdown: float
    max_drawdown: float
    var_95: float
    volatility: float
    sharpe_ratio: float
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class Portfolio:
    """Tracks portfolio state: capital, positions, equity history."""

    capital: float
    positions: dict[str, float] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    peak_equity: float = 0.0

    def __post_init__(self):
        if not self.equity_curve:
            self.equity_curve = [self.capital]
        self.peak_equity = max(self.peak_equity, self.capital)

    @property
    def total_equity(self) -> float:
        return self.capital + sum(self.positions.values())

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.total_equity) / self.peak_equity


class RiskManager:
    """Position sizing and drawdown control via Kelly + vol scaling."""

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_portfolio_risk: float = 0.20,
        max_drawdown_pct: float = 0.15,
        risk_free_rate: float = 0.05,
        kelly_fraction: float = 0.25,
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_free_rate = risk_free_rate
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        price: float,
        volatility: float,
        portfolio: Portfolio,
        win_rate: float | None = None,
        win_loss_ratio: float | None = None,
    ) -> PositionSize:
        """Size a position using min(Kelly, vol-scaled, max limit)."""
        total_equity = portfolio.total_equity
        max_notional = total_equity * self.max_position_pct

        if win_rate is not None and win_loss_ratio is not None:
            kelly_pct = self._kelly_criterion(win_rate, win_loss_ratio)
            kelly_notional = total_equity * kelly_pct * self.kelly_fraction
        else:
            kelly_notional = max_notional * 0.5

        vol_scalar = self._volatility_scalar(volatility)
        vol_notional = max_notional * vol_scalar

        base_notional = min(max_notional, kelly_notional, vol_notional)
        signal_adjusted = base_notional * abs(signal_strength)

        drawdown_factor = self._drawdown_factor(portfolio.current_drawdown)
        final_notional = signal_adjusted * drawdown_factor

        shares = int(final_notional / price) if price > 0 else 0
        actual_notional = shares * price
        weight = actual_notional / total_equity if total_equity > 0 else 0

        risk_score = self._calculate_risk_score(
            weight, volatility, portfolio.current_drawdown
        )

        return PositionSize(
            symbol=symbol,
            shares=shares,
            notional=actual_notional,
            weight=weight,
            risk_score=risk_score,
        )

    def _kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """Kelly optimal bet fraction: f* = (p*b - q) / b."""
        if win_loss_ratio <= 0:
            return 0.0
        p = max(0, min(1, win_rate))
        q = 1 - p
        b = win_loss_ratio
        kelly = (p * b - q) / b
        return max(0, min(kelly, 1.0))

    def _volatility_scalar(self, volatility: float, target_vol: float = 0.15) -> float:
        """Inverse vol scaling: higher vol -> smaller position."""
        if volatility <= 0:
            return 1.0
        scalar = target_vol / volatility
        return max(0.1, min(scalar, 2.0))

    def _drawdown_factor(self, current_drawdown: float) -> float:
        """Linearly reduce position size as drawdown approaches the limit."""
        if current_drawdown <= 0:
            return 1.0
        if current_drawdown >= self.max_drawdown_pct:
            return 0.0
        drawdown_ratio = current_drawdown / self.max_drawdown_pct
        return 1.0 - (0.75 * drawdown_ratio)

    def _calculate_risk_score(
        self, weight: float, volatility: float, drawdown: float
    ) -> float:
        """Composite risk score: concentration + volatility + drawdown."""
        concentration_risk = weight / self.max_position_pct
        vol_risk = volatility / 0.20
        dd_risk = drawdown / self.max_drawdown_pct
        risk_score = 0.3 * concentration_risk + 0.4 * vol_risk + 0.3 * dd_risk
        return max(0, min(1, risk_score))

    def calculate_portfolio_risk(
        self, returns: np.ndarray, equity_curve: np.ndarray
    ) -> RiskMetrics:
        """Compute drawdown, VaR, volatility, and Sharpe from return/equity arrays."""
        if len(returns) < 2:
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, RiskLevel.LOW)

        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        current_dd = drawdowns[-1]
        max_dd = np.max(drawdowns)

        var_95 = np.percentile(returns, 5)

        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = (
            (np.mean(excess_returns) / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
        )

        risk_level = self._assess_risk_level(current_dd, max_dd, annual_vol)

        return RiskMetrics(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            var_95=var_95,
            volatility=annual_vol,
            sharpe_ratio=sharpe,
            risk_level=risk_level,
        )

    def _assess_risk_level(
        self, current_dd: float, max_dd: float, volatility: float
    ) -> RiskLevel:
        """Map risk metrics to a categorical level."""
        risk_score = (
            (current_dd / self.max_drawdown_pct) * 0.4
            + (max_dd / (self.max_drawdown_pct * 1.5)) * 0.3
            + (volatility / 0.30) * 0.3
        )
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.85:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def should_reduce_risk(self, portfolio: Portfolio) -> bool:
        """True if drawdown is at 75%+ of the max allowed."""
        return portfolio.current_drawdown >= (self.max_drawdown_pct * 0.75)

    def get_risk_budget(self, portfolio: Portfolio) -> float:
        """Remaining risk budget as a fraction (0 = fully used, 1 = untouched)."""
        used_risk = portfolio.current_drawdown / self.max_drawdown_pct
        return max(0, 1 - used_risk)
