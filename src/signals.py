"""
Statistical Signal Generation Engine

This module implements various technical indicators and statistical signals
used in quantitative trading strategies. All calculations are vectorized
using NumPy for performance.

Key Concepts:
    - Z-Score for mean reversion
    - RSI for momentum
    - Bollinger Bands for volatility
    - MACD for trend following
    - Signal combination and normalization
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class SignalType(Enum):
    """Types of trading signals."""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"


@dataclass
class Signal:
    """
    Container for a trading signal.
    
    Attributes:
        name: Signal identifier
        values: Array of signal values over time
        signal_type: Category of signal
        strength: Overall signal strength (0 to 1)
    """
    name: str
    values: np.ndarray
    signal_type: SignalType
    strength: float
    
    def get_current(self) -> float:
        """Get the most recent signal value."""
        return self.values[-1] if len(self.values) > 0 else 0.0
    
    def get_direction(self) -> int:
        """
        Get trading direction based on current signal.
        
        Returns:
            1 for long, -1 for short, 0 for neutral
        """
        current = self.get_current()
        if current > 0.5:
            return 1
        elif current < -0.5:
            return -1
        return 0


class SignalEngine:
    """
    Statistical signal generation engine.
    
    This class implements various technical indicators used in quantitative
    trading. All methods are vectorized for performance and return normalized
    signals in the range [-1, 1].
    
    Signals Implemented:
        - Z-Score: Mean reversion based on rolling statistics
        - RSI: Relative Strength Index for overbought/oversold
        - Bollinger Bands: Volatility-adjusted price channels
        - MACD: Moving Average Convergence Divergence
    
    Example:
        >>> engine = SignalEngine()
        >>> prices = np.array([100, 101, 99, 102, 98, 103])
        >>> z_signal = engine.calculate_zscore(prices, window=3)
        >>> print(f"Current Z-Score: {z_signal.get_current():.2f}")
    """
    
    def __init__(self, default_window: int = 20):
        """
        Initialize the signal engine.
        
        Args:
            default_window: Default lookback period for calculations
        """
        self.default_window = default_window
    
    def calculate_zscore(
        self,
        prices: np.ndarray,
        window: Optional[int] = None
    ) -> Signal:
        """
        Calculate Z-Score signal for mean reversion trading.
        
        The Z-Score measures how many standard deviations the current
        price is from its rolling mean:
        
            z = (price - rolling_mean) / rolling_std
        
        Trading Logic:
            - z < -2: Price unusually low → BUY signal
            - z > +2: Price unusually high → SELL signal
            - |z| < 1: Neutral zone
        
        This is the foundation of statistical arbitrage and pairs trading.
        
        Args:
            prices: Array of price data
            window: Lookback period for rolling statistics
        
        Returns:
            Signal object with normalized values in [-1, 1]
        
        Mathematical Background:
            Under mean reversion assumption, prices oscillate around a
            mean value. Extreme deviations are expected to reverse.
            The Z-score quantifies "extremeness" in standard deviation units.
        """
        window = window or self.default_window
        
        if len(prices) < window:
            return Signal(
                name="zscore",
                values=np.zeros(len(prices)),
                signal_type=SignalType.MEAN_REVERSION,
                strength=0.0
            )
        
        # Calculate rolling statistics (vectorized)
        rolling_mean = self._rolling_mean(prices, window)
        rolling_std = self._rolling_std(prices, window)
        
        # Avoid division by zero
        rolling_std = np.maximum(rolling_std, 1e-10)
        
        # Calculate raw Z-scores
        zscore = (prices - rolling_mean) / rolling_std
        
        # Normalize to [-1, 1] using tanh
        # tanh naturally squashes values and handles extreme outliers
        normalized = np.tanh(zscore / 2)
        
        # For mean reversion, we want to BUY when price is LOW (negative z)
        # So we INVERT the signal: negative z → positive signal
        signal_values = -normalized
        
        # Calculate signal strength based on recent volatility
        strength = min(1.0, np.std(zscore[-window:]) / 2)
        
        return Signal(
            name="zscore",
            values=signal_values,
            signal_type=SignalType.MEAN_REVERSION,
            strength=strength
        )
    
    def calculate_rsi(
        self,
        prices: np.ndarray,
        window: int = 14
    ) -> Signal:
        """
        Calculate Relative Strength Index (RSI) signal.
        
        RSI measures the speed and magnitude of recent price changes
        to evaluate overbought or oversold conditions:
        
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss
        
        Trading Logic:
            - RSI < 30: Oversold → BUY signal
            - RSI > 70: Overbought → SELL signal
            - 30 < RSI < 70: Neutral
        
        Args:
            prices: Array of price data
            window: RSI calculation period (typically 14)
        
        Returns:
            Signal object with values in [-1, 1]
        
        Note:
            RSI is a bounded oscillator (0-100), making it useful for
            identifying potential reversal points.
        """
        if len(prices) < window + 1:
            return Signal(
                name="rsi",
                values=np.zeros(len(prices)),
                signal_type=SignalType.MOMENTUM,
                strength=0.0
            )
        
        # Calculate price changes
        delta = np.diff(prices, prepend=prices[0])
        
        # Separate gains and losses
        gains = np.maximum(delta, 0)
        losses = np.abs(np.minimum(delta, 0))
        
        # Calculate smoothed averages (Wilder's smoothing)
        avg_gain = self._exponential_smoothing(gains, window)
        avg_loss = self._exponential_smoothing(losses, window)
        
        # Avoid division by zero
        avg_loss = np.maximum(avg_loss, 1e-10)
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize RSI to [-1, 1]
        # RSI 50 → 0, RSI 0 → -1, RSI 100 → +1
        # Then invert for trading signal (low RSI = buy = positive signal)
        normalized = -((rsi - 50) / 50)
        
        # Signal strength based on how extreme RSI is
        strength = np.mean(np.abs(rsi[-window:] - 50)) / 50
        
        return Signal(
            name="rsi",
            values=normalized,
            signal_type=SignalType.MOMENTUM,
            strength=min(1.0, strength)
        )
    
    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[Signal, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands and %B signal.
        
        Bollinger Bands consist of:
            - Middle Band: Simple Moving Average
            - Upper Band: SMA + (num_std × standard deviation)
            - Lower Band: SMA - (num_std × standard deviation)
        
        %B measures where price is relative to the bands:
            %B = (Price - Lower Band) / (Upper Band - Lower Band)
        
        Trading Logic:
            - %B < 0: Price below lower band → BUY signal
            - %B > 1: Price above upper band → SELL signal
            - %B ≈ 0.5: Price at middle band → Neutral
        
        Args:
            prices: Array of price data
            window: Period for SMA and standard deviation
            num_std: Number of standard deviations for bands
        
        Returns:
            Tuple of (Signal, upper_band, middle_band, lower_band)
        """
        if len(prices) < window:
            zeros = np.zeros(len(prices))
            return (
                Signal("bollinger", zeros, SignalType.VOLATILITY, 0.0),
                prices.copy(),
                prices.copy(),
                prices.copy()
            )
        
        # Calculate bands
        middle_band = self._rolling_mean(prices, window)
        rolling_std = self._rolling_std(prices, window)
        
        upper_band = middle_band + (num_std * rolling_std)
        lower_band = middle_band - (num_std * rolling_std)
        
        # Calculate %B
        band_width = upper_band - lower_band
        band_width = np.maximum(band_width, 1e-10)  # Avoid division by zero
        
        percent_b = (prices - lower_band) / band_width
        
        # Normalize to [-1, 1] and invert for trading signal
        # %B < 0.5 → buy signal (positive), %B > 0.5 → sell signal (negative)
        normalized = -(percent_b - 0.5) * 2
        normalized = np.clip(normalized, -1, 1)
        
        # Signal strength based on band width (volatility)
        strength = np.mean(rolling_std[-window:] / prices[-window:]) * 10
        
        signal = Signal(
            name="bollinger",
            values=normalized,
            signal_type=SignalType.VOLATILITY,
            strength=min(1.0, strength)
        )
        
        return signal, upper_band, middle_band, lower_band
    
    def calculate_macd(
        self,
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Signal:
        """
        Calculate MACD (Moving Average Convergence Divergence) signal.
        
        MACD Components:
            - MACD Line: Fast EMA - Slow EMA
            - Signal Line: EMA of MACD Line
            - Histogram: MACD Line - Signal Line
        
        Trading Logic:
            - MACD crosses above Signal: BUY
            - MACD crosses below Signal: SELL
            - Histogram magnitude indicates trend strength
        
        Args:
            prices: Array of price data
            fast_period: Period for fast EMA (typically 12)
            slow_period: Period for slow EMA (typically 26)
            signal_period: Period for signal line EMA (typically 9)
        
        Returns:
            Signal object based on MACD histogram
        
        Note:
            MACD is a trend-following indicator that shows the relationship
            between two moving averages of price.
        """
        if len(prices) < slow_period:
            return Signal(
                name="macd",
                values=np.zeros(len(prices)),
                signal_type=SignalType.TREND,
                strength=0.0
            )
        
        # Calculate EMAs
        fast_ema = self._ema(prices, fast_period)
        slow_ema = self._ema(prices, slow_period)
        
        # MACD Line
        macd_line = fast_ema - slow_ema
        
        # Signal Line
        signal_line = self._ema(macd_line, signal_period)
        
        # Histogram (this is our trading signal)
        histogram = macd_line - signal_line
        
        # Normalize histogram to [-1, 1]
        # Use rolling max for adaptive scaling
        window = slow_period
        rolling_max = self._rolling_max(np.abs(histogram), window)
        rolling_max = np.maximum(rolling_max, 1e-10)
        
        normalized = histogram / rolling_max
        normalized = np.clip(normalized, -1, 1)
        
        # Signal strength based on trend clarity
        strength = np.mean(np.abs(normalized[-signal_period:])) 
        
        return Signal(
            name="macd",
            values=normalized,
            signal_type=SignalType.TREND,
            strength=min(1.0, strength)
        )
    
    def combine_signals(
        self,
        signals: Dict[str, Signal],
        weights: Optional[Dict[str, float]] = None
    ) -> Signal:
        """
        Combine multiple signals into a composite signal.
        
        This implements a simple weighted average of signals.
        More sophisticated combination methods (e.g., ML-based)
        could be implemented here.
        
        Args:
            signals: Dictionary of signal name to Signal object
            weights: Optional custom weights (default: equal weight)
        
        Returns:
            Combined Signal object
        
        Example:
            >>> combined = engine.combine_signals({
            ...     'zscore': z_signal,
            ...     'rsi': rsi_signal,
            ...     'macd': macd_signal
            ... }, weights={'zscore': 0.5, 'rsi': 0.3, 'macd': 0.2})
        """
        if not signals:
            return Signal(
                name="combined",
                values=np.array([0.0]),
                signal_type=SignalType.MEAN_REVERSION,
                strength=0.0
            )
        
        # Default to equal weights
        if weights is None:
            weights = {name: 1.0 / len(signals) for name in signals}
        
        # Normalize weights
        total_weight = sum(weights.get(name, 0) for name in signals)
        if total_weight == 0:
            total_weight = 1.0
        
        # Find common length
        min_length = min(len(s.values) for s in signals.values())
        
        # Weighted combination
        combined_values = np.zeros(min_length)
        combined_strength = 0.0
        
        for name, signal in signals.items():
            weight = weights.get(name, 0) / total_weight
            combined_values += weight * signal.values[-min_length:]
            combined_strength += weight * signal.strength
        
        # Clip to valid range
        combined_values = np.clip(combined_values, -1, 1)
        
        return Signal(
            name="combined",
            values=combined_values,
            signal_type=SignalType.MEAN_REVERSION,
            strength=combined_strength
        )
    
    # =========================================================================
    # Helper Methods (Vectorized Operations)
    # =========================================================================
    
    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling mean using cumsum trick for O(n) complexity.
        
        This is faster than naive iteration for large arrays.
        """
        n = len(data)
        result = np.empty(n)
        
        # Expanding mean for first window-1 elements
        cumsum = np.cumsum(data)
        result[:window] = cumsum[:window] / np.arange(1, window + 1)
        
        # Rolling mean for rest
        if n > window:
            result[window:] = (cumsum[window:] - cumsum[:-window]) / window
        
        return result
    
    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        rolling_mean = SignalEngine._rolling_mean(data, window)
        rolling_sq_mean = SignalEngine._rolling_mean(data ** 2, window)
        
        # Var = E[X^2] - E[X]^2
        variance = rolling_sq_mean - rolling_mean ** 2
        variance = np.maximum(variance, 0)  # Handle numerical issues
        
        return np.sqrt(variance)
    
    @staticmethod
    def _rolling_max(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling maximum."""
        result = np.empty_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.max(data[start:i + 1])
        return result
    
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        EMA gives more weight to recent prices:
            EMA_t = α × Price_t + (1 - α) × EMA_{t-1}
            where α = 2 / (period + 1)
        """
        alpha = 2 / (period + 1)
        result = np.empty_like(data)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def _exponential_smoothing(data: np.ndarray, period: int) -> np.ndarray:
        """
        Wilder's exponential smoothing (used in RSI).
        
        Different from standard EMA:
            α = 1 / period (instead of 2 / (period + 1))
        """
        alpha = 1 / period
        result = np.empty_like(data)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result


# Example usage
if __name__ == "__main__":
    # Generate sample price data
    np.random.seed(42)
    n_days = 100
    
    # Simulate mean-reverting price (Ornstein-Uhlenbeck process)
    prices = np.zeros(n_days)
    prices[0] = 100
    theta = 0.1  # Mean reversion speed
    mu = 100     # Long-term mean
    sigma = 2    # Volatility
    
    for i in range(1, n_days):
        prices[i] = (
            prices[i-1] + 
            theta * (mu - prices[i-1]) + 
            sigma * np.random.randn()
        )
    
    # Calculate signals
    engine = SignalEngine()
    
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
    })
    
    print("Signal Analysis")
    print("=" * 50)
    print(f"Z-Score:    {zscore.get_current():+.3f} (strength: {zscore.strength:.2f})")
    print(f"RSI:        {rsi.get_current():+.3f} (strength: {rsi.strength:.2f})")
    print(f"MACD:       {macd.get_current():+.3f} (strength: {macd.strength:.2f})")
    print(f"Bollinger:  {bollinger.get_current():+.3f} (strength: {bollinger.strength:.2f})")
    print(f"Combined:   {combined.get_current():+.3f} (strength: {combined.strength:.2f})")
    print(f"\nTrading Direction: {combined.get_direction()} (1=Long, -1=Short, 0=Neutral)")
