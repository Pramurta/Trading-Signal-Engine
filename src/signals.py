"""Signal generation engine — Z-Score, RSI, Bollinger Bands, MACD."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalType(Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"


@dataclass
class Signal:
    """A trading signal with time series values normalized to [-1, 1]."""

    name: str
    values: np.ndarray
    signal_type: SignalType
    strength: float  # 0 to 1

    def get_current(self) -> float:
        """Most recent signal value."""
        return self.values[-1] if len(self.values) > 0 else 0.0

    def get_direction(self) -> int:
        """1 for long, -1 for short, 0 for neutral (threshold: +/-0.5)."""
        current = self.get_current()
        if current > 0.5:
            return 1
        elif current < -0.5:
            return -1
        return 0


class SignalEngine:
    """Computes technical indicators and combines them into trading signals."""

    def __init__(self, default_window: int = 20):
        self.default_window = default_window

    def calculate_zscore(self, prices: np.ndarray, window: int | None = None) -> Signal:
        """Z-Score mean reversion signal. Inverted for mean reversion."""
        window = window or self.default_window

        if len(prices) < window:
            return Signal(
                "zscore",
                np.zeros(len(prices)),
                SignalType.MEAN_REVERSION,
                0.0,
            )

        rolling_mean = self._rolling_mean(prices, window)
        rolling_std = self._rolling_std(prices, window)
        rolling_std = np.maximum(rolling_std, 1e-10)

        zscore = (prices - rolling_mean) / rolling_std
        # tanh normalization to [-1, 1], inverted for mean reversion
        signal_values = -np.tanh(zscore / 2)

        strength = min(1.0, np.std(zscore[-window:]) / 2)

        return Signal("zscore", signal_values, SignalType.MEAN_REVERSION, strength)

    def calculate_rsi(self, prices: np.ndarray, window: int = 14) -> Signal:
        """RSI momentum signal. Inverted: oversold -> buy signal (positive)."""
        if len(prices) < window + 1:
            return Signal("rsi", np.zeros(len(prices)), SignalType.MOMENTUM, 0.0)

        delta = np.diff(prices, prepend=prices[0])
        gains = np.maximum(delta, 0)
        losses = np.abs(np.minimum(delta, 0))

        avg_gain = self._exponential_smoothing(gains, window)
        avg_loss = self._exponential_smoothing(losses, window)
        avg_loss = np.maximum(avg_loss, 1e-10)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Normalize to [-1, 1] and invert (low RSI = buy = positive)
        normalized = -((rsi - 50) / 50)

        strength = np.mean(np.abs(rsi[-window:] - 50)) / 50

        return Signal("rsi", normalized, SignalType.MOMENTUM, min(1.0, strength))

    def calculate_bollinger_bands(
        self, prices: np.ndarray, window: int = 20, num_std: float = 2.0
    ) -> tuple[Signal, np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger %B signal and bands. Returns (signal, upper, middle, lower)."""
        if len(prices) < window:
            zeros = np.zeros(len(prices))
            return (
                Signal("bollinger", zeros, SignalType.VOLATILITY, 0.0),
                prices.copy(),
                prices.copy(),
                prices.copy(),
            )

        middle_band = self._rolling_mean(prices, window)
        rolling_std = self._rolling_std(prices, window)

        upper_band = middle_band + (num_std * rolling_std)
        lower_band = middle_band - (num_std * rolling_std)

        band_width = np.maximum(upper_band - lower_band, 1e-10)
        percent_b = (prices - lower_band) / band_width

        # Normalize and invert: below middle -> buy signal
        normalized = np.clip(-(percent_b - 0.5) * 2, -1, 1)

        strength = np.mean(rolling_std[-window:] / prices[-window:]) * 10

        signal = Signal(
            "bollinger",
            normalized,
            SignalType.VOLATILITY,
            min(1.0, strength),
        )
        return signal, upper_band, middle_band, lower_band

    def calculate_macd(
        self,
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Signal:
        """MACD trend signal based on the histogram (MACD line - signal line)."""
        if len(prices) < slow_period:
            return Signal("macd", np.zeros(len(prices)), SignalType.TREND, 0.0)

        fast_ema = self._ema(prices, fast_period)
        slow_ema = self._ema(prices, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = self._ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        # Adaptive normalization using rolling max
        rolling_max = self._rolling_max(np.abs(histogram), slow_period)
        rolling_max = np.maximum(rolling_max, 1e-10)
        normalized = np.clip(histogram / rolling_max, -1, 1)

        strength = np.mean(np.abs(normalized[-signal_period:]))

        return Signal("macd", normalized, SignalType.TREND, min(1.0, strength))

    def combine_signals(
        self, signals: dict[str, Signal], weights: dict[str, float] | None = None
    ) -> Signal:
        """Weighted average of multiple signals."""
        if not signals:
            return Signal("combined", np.array([0.0]), SignalType.MEAN_REVERSION, 0.0)

        if weights is None:
            weights = {name: 1.0 / len(signals) for name in signals}

        total_weight = sum(weights.get(name, 0) for name in signals)
        if total_weight == 0:
            total_weight = 1.0

        min_length = min(len(s.values) for s in signals.values())

        combined_values = np.zeros(min_length)
        combined_strength = 0.0

        for name, signal in signals.items():
            weight = weights.get(name, 0) / total_weight
            combined_values += weight * signal.values[-min_length:]
            combined_strength += weight * signal.strength

        combined_values = np.clip(combined_values, -1, 1)

        return Signal(
            "combined",
            combined_values,
            SignalType.MEAN_REVERSION,
            combined_strength,
        )

    # --- Vectorized helpers ---

    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean using cumsum for O(n) complexity."""
        n = len(data)
        result = np.empty(n)
        cumsum = np.cumsum(data)
        result[:window] = cumsum[:window] / np.arange(1, window + 1)
        if n > window:
            result[window:] = (cumsum[window:] - cumsum[:-window]) / window
        return result

    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation via Var = E[X^2] - E[X]^2."""
        rolling_mean = SignalEngine._rolling_mean(data, window)
        rolling_sq_mean = SignalEngine._rolling_mean(data**2, window)
        variance = np.maximum(rolling_sq_mean - rolling_mean**2, 0)
        return np.sqrt(variance)  # type: ignore[no-any-return]

    @staticmethod
    def _rolling_max(data: np.ndarray, window: int) -> np.ndarray:
        """Rolling maximum over a window."""
        result = np.empty_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.max(data[start : i + 1])
        return result

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average with alpha = 2 / (period + 1)."""
        alpha = 2 / (period + 1)
        result = np.empty_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _exponential_smoothing(data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing (alpha = 1/period), used for RSI."""
        alpha = 1 / period
        result = np.empty_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result
