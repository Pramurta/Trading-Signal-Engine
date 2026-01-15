"""
Risk Management Module

This module implements position sizing and risk control mechanisms
essential for systematic trading. It includes:
    - Kelly Criterion for optimal position sizing
    - Volatility-adjusted position scaling
    - Maximum drawdown controls
    - Portfolio-level risk limits

Key Concepts:
    - Risk-adjusted returns
    - Position sizing under uncertainty
    - Drawdown management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

import numpy as np


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionSize:
    """
    Recommended position size with risk metrics.
    
    Attributes:
        symbol: Trading symbol
        shares: Number of shares to trade
        notional: Dollar value of position
        weight: Portfolio weight (0 to 1)
        risk_score: Risk assessment (0 to 1)
    """
    symbol: str
    shares: int
    notional: float
    weight: float
    risk_score: float
    
    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.shares:,} shares "
            f"(${self.notional:,.0f}, {self.weight:.1%} weight)"
        )


@dataclass 
class RiskMetrics:
    """
    Portfolio risk metrics.
    
    Attributes:
        current_drawdown: Current drawdown from peak
        max_drawdown: Maximum historical drawdown
        var_95: Value at Risk at 95% confidence
        volatility: Annualized portfolio volatility
        sharpe_ratio: Risk-adjusted return metric
    """
    current_drawdown: float
    max_drawdown: float
    var_95: float
    volatility: float
    sharpe_ratio: float
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class Portfolio:
    """
    Track portfolio state and history.
    
    Attributes:
        capital: Current available capital
        positions: Dictionary of symbol to position size
        equity_curve: Historical equity values
        peak_equity: Highest equity value achieved
    """
    capital: float
    positions: Dict[str, float] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    peak_equity: float = 0.0
    
    def __post_init__(self):
        if not self.equity_curve:
            self.equity_curve = [self.capital]
        self.peak_equity = max(self.peak_equity, self.capital)
    
    @property
    def total_equity(self) -> float:
        """Total portfolio value including positions."""
        position_value = sum(self.positions.values())
        return self.capital + position_value
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.total_equity) / self.peak_equity


class RiskManager:
    """
    Risk management system for position sizing and drawdown control.
    
    This class implements:
        1. Kelly Criterion for optimal bet sizing
        2. Volatility-scaled position sizing
        3. Maximum drawdown limits
        4. Portfolio-level risk monitoring
    
    The Kelly Criterion finds the optimal fraction of capital to risk:
        f* = (p × b - q) / b
        
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio (average win / average loss)
    
    Example:
        >>> risk_mgr = RiskManager(max_position_pct=0.1)
        >>> size = risk_mgr.calculate_position_size(
        ...     symbol='AAPL',
        ...     signal_strength=0.7,
        ...     price=150.0,
        ...     volatility=0.25,
        ...     portfolio=Portfolio(capital=100000)
        ... )
        >>> print(size)
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_portfolio_risk: float = 0.20,
        max_drawdown_pct: float = 0.15,
        risk_free_rate: float = 0.05,
        kelly_fraction: float = 0.25  # Fractional Kelly for safety
    ):
        """
        Initialize risk manager with risk limits.
        
        Args:
            max_position_pct: Maximum single position as % of portfolio
            max_portfolio_risk: Maximum total portfolio risk
            max_drawdown_pct: Maximum allowed drawdown before reducing risk
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
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
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate optimal position size using multiple methods.
        
        This combines:
            1. Kelly Criterion (if win rate provided)
            2. Volatility scaling (inverse relationship)
            3. Signal strength adjustment
            4. Drawdown-based risk reduction
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal confidence (0 to 1)
            price: Current price per share
            volatility: Annualized volatility of the asset
            portfolio: Current portfolio state
            win_rate: Historical win probability (optional)
            win_loss_ratio: Avg win / avg loss (optional)
        
        Returns:
            PositionSize with recommended trade size
        
        Position Sizing Logic:
            base_size = min(kelly_size, volatility_size, max_position)
            adjusted_size = base_size × signal_strength × drawdown_factor
        """
        total_equity = portfolio.total_equity
        
        # Start with maximum allowed position
        max_notional = total_equity * self.max_position_pct
        
        # Method 1: Kelly Criterion (if we have statistics)
        if win_rate is not None and win_loss_ratio is not None:
            kelly_pct = self._kelly_criterion(win_rate, win_loss_ratio)
            kelly_notional = total_equity * kelly_pct * self.kelly_fraction
        else:
            # Default to conservative estimate
            kelly_notional = max_notional * 0.5
        
        # Method 2: Volatility-based sizing
        # Higher volatility → smaller position
        vol_scalar = self._volatility_scalar(volatility)
        vol_notional = max_notional * vol_scalar
        
        # Take the minimum of all methods (conservative)
        base_notional = min(max_notional, kelly_notional, vol_notional)
        
        # Adjust for signal strength
        signal_adjusted = base_notional * abs(signal_strength)
        
        # Adjust for current drawdown (reduce risk if in drawdown)
        drawdown_factor = self._drawdown_factor(portfolio.current_drawdown)
        final_notional = signal_adjusted * drawdown_factor
        
        # Calculate shares
        shares = int(final_notional / price) if price > 0 else 0
        actual_notional = shares * price
        
        # Calculate portfolio weight
        weight = actual_notional / total_equity if total_equity > 0 else 0
        
        # Risk score (0 = low risk, 1 = high risk)
        risk_score = self._calculate_risk_score(
            weight, volatility, portfolio.current_drawdown
        )
        
        return PositionSize(
            symbol=symbol,
            shares=shares,
            notional=actual_notional,
            weight=weight,
            risk_score=risk_score
        )
    
    def _kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion optimal bet fraction.
        
        Formula:
            f* = (p × b - q) / b
            
        where:
            p = win probability
            q = loss probability (1 - p)
            b = win/loss ratio
        
        Args:
            win_rate: Probability of winning (0 to 1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Optimal fraction of capital to bet
        
        Note:
            Full Kelly is often too aggressive in practice.
            We use fractional Kelly (typically 0.25-0.5) for safety.
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        p = max(0, min(1, win_rate))  # Clamp to [0, 1]
        q = 1 - p
        b = win_loss_ratio
        
        kelly = (p * b - q) / b
        
        # Kelly can be negative (don't bet) or > 1 (use leverage)
        # We clamp to reasonable bounds
        return max(0, min(kelly, 1.0))
    
    def _volatility_scalar(
        self,
        volatility: float,
        target_vol: float = 0.15
    ) -> float:
        """
        Calculate position scalar based on volatility.
        
        Inverse relationship: higher vol → smaller position
        This implements "risk parity" style sizing.
        
        Args:
            volatility: Asset's annualized volatility
            target_vol: Target portfolio volatility
        
        Returns:
            Scalar to apply to position size (0 to 1)
        """
        if volatility <= 0:
            return 1.0
        
        # Scalar = target / actual (inverse relationship)
        scalar = target_vol / volatility
        
        # Clamp to reasonable range
        return max(0.1, min(scalar, 2.0))
    
    def _drawdown_factor(self, current_drawdown: float) -> float:
        """
        Reduce position size based on current drawdown.
        
        This implements a simple risk-off mechanism:
            - No drawdown: full size
            - 50% of max drawdown: 75% size
            - At max drawdown: 25% size
            - Beyond max: 0% (stop trading)
        
        Args:
            current_drawdown: Current drawdown as fraction (0 to 1)
        
        Returns:
            Factor to multiply position size by
        """
        if current_drawdown <= 0:
            return 1.0
        
        if current_drawdown >= self.max_drawdown_pct:
            return 0.0  # Stop trading
        
        # Linear reduction
        drawdown_ratio = current_drawdown / self.max_drawdown_pct
        return 1.0 - (0.75 * drawdown_ratio)
    
    def _calculate_risk_score(
        self,
        weight: float,
        volatility: float,
        drawdown: float
    ) -> float:
        """
        Calculate composite risk score for a position.
        
        Combines:
            - Position concentration risk
            - Asset volatility risk
            - Portfolio drawdown risk
        
        Args:
            weight: Position weight in portfolio
            volatility: Asset volatility
            drawdown: Current portfolio drawdown
        
        Returns:
            Risk score from 0 (low) to 1 (high)
        """
        # Concentration risk (weight / max allowed)
        concentration_risk = weight / self.max_position_pct
        
        # Volatility risk (normalized around 20% vol)
        vol_risk = volatility / 0.20
        
        # Drawdown risk
        dd_risk = drawdown / self.max_drawdown_pct
        
        # Weighted average
        risk_score = (
            0.3 * concentration_risk +
            0.4 * vol_risk +
            0.3 * dd_risk
        )
        
        return max(0, min(1, risk_score))
    
    def calculate_portfolio_risk(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Metrics calculated:
            - Current and maximum drawdown
            - Value at Risk (VaR) at 95% confidence
            - Annualized volatility
            - Sharpe ratio
        
        Args:
            returns: Array of portfolio returns
            equity_curve: Array of portfolio equity values
        
        Returns:
            RiskMetrics dataclass with all metrics
        """
        if len(returns) < 2:
            return RiskMetrics(
                current_drawdown=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                risk_level=RiskLevel.LOW
            )
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        current_dd = drawdowns[-1]
        max_dd = np.max(drawdowns)
        
        # Value at Risk (historical method)
        var_95 = np.percentile(returns, 5)
        
        # Annualized volatility
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = (np.mean(excess_returns) / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
        
        # Determine risk level
        risk_level = self._assess_risk_level(current_dd, max_dd, annual_vol)
        
        return RiskMetrics(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            var_95=var_95,
            volatility=annual_vol,
            sharpe_ratio=sharpe,
            risk_level=risk_level
        )
    
    def _assess_risk_level(
        self,
        current_dd: float,
        max_dd: float,
        volatility: float
    ) -> RiskLevel:
        """Assess overall risk level based on metrics."""
        risk_score = (
            (current_dd / self.max_drawdown_pct) * 0.4 +
            (max_dd / (self.max_drawdown_pct * 1.5)) * 0.3 +
            (volatility / 0.30) * 0.3
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
        """
        Check if portfolio is in drawdown danger zone.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            True if risk should be reduced
        """
        return portfolio.current_drawdown >= (self.max_drawdown_pct * 0.75)
    
    def get_risk_budget(self, portfolio: Portfolio) -> float:
        """
        Calculate remaining risk budget.
        
        This tells us how much additional risk we can take
        before hitting our maximum drawdown limit.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            Remaining risk budget as percentage
        """
        used_risk = portfolio.current_drawdown / self.max_drawdown_pct
        return max(0, 1 - used_risk)


# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_mgr = RiskManager(
        max_position_pct=0.10,
        max_drawdown_pct=0.15,
        kelly_fraction=0.25
    )
    
    # Create portfolio
    portfolio = Portfolio(capital=100_000)
    
    # Calculate position size for a trade
    position = risk_mgr.calculate_position_size(
        symbol="AAPL",
        signal_strength=0.75,
        price=175.00,
        volatility=0.25,
        portfolio=portfolio,
        win_rate=0.55,
        win_loss_ratio=1.5
    )
    
    print("Position Sizing Example")
    print("=" * 50)
    print(f"Portfolio Capital: ${portfolio.capital:,.0f}")
    print(f"Signal Strength: 0.75")
    print(f"Asset Volatility: 25%")
    print(f"\nRecommended Position:")
    print(f"  {position}")
    print(f"  Risk Score: {position.risk_score:.2f}")
    
    # Show Kelly calculation
    print("\nKelly Criterion:")
    kelly = risk_mgr._kelly_criterion(0.55, 1.5)
    print(f"  Win Rate: 55%")
    print(f"  Win/Loss Ratio: 1.5x")
    print(f"  Full Kelly: {kelly:.1%}")
    print(f"  Quarter Kelly: {kelly * 0.25:.1%}")
    
    # Simulate drawdown scenario
    print("\nDrawdown Scenario:")
    portfolio_dd = Portfolio(capital=85_000)
    portfolio_dd.peak_equity = 100_000
    print(f"  Peak Equity: ${portfolio_dd.peak_equity:,.0f}")
    print(f"  Current Equity: ${portfolio_dd.capital:,.0f}")
    print(f"  Current Drawdown: {portfolio_dd.current_drawdown:.1%}")
    print(f"  Should Reduce Risk: {risk_mgr.should_reduce_risk(portfolio_dd)}")
    print(f"  Risk Budget Remaining: {risk_mgr.get_risk_budget(portfolio_dd):.1%}")
