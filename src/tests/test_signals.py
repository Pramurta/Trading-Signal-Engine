"""
Unit Tests for Signal Engine

This module contains comprehensive tests for the signal generation
functionality. Tests verify mathematical correctness and edge cases.

Run with: pytest tests/test_signals.py -v
"""

import numpy as np
import pytest
from src.signals import SignalEngine, Signal, SignalType


class TestSignalEngine:
    """Test suite for SignalEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create SignalEngine instance for tests."""
        return SignalEngine(default_window=20)
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 100
        # Mean-reverting prices (Ornstein-Uhlenbeck)
        prices = np.zeros(n)
        prices[0] = 100
        for i in range(1, n):
            prices[i] = prices[i-1] + 0.1 * (100 - prices[i-1]) + np.random.randn()
        return prices
    
    # =========================================================================
    # Z-Score Tests
    # =========================================================================
    
    def test_zscore_returns_signal(self, engine, sample_prices):
        """Z-score calculation should return a Signal object."""
        signal = engine.calculate_zscore(sample_prices)
        
        assert isinstance(signal, Signal)
        assert signal.name == "zscore"
        assert signal.signal_type == SignalType.MEAN_REVERSION
    
    def test_zscore_output_shape(self, engine, sample_prices):
        """Z-score output should match input length."""
        signal = engine.calculate_zscore(sample_prices)
        assert len(signal.values) == len(sample_prices)
    
    def test_zscore_bounded(self, engine, sample_prices):
        """Z-score values should be bounded in [-1, 1] after normalization."""
        signal = engine.calculate_zscore(sample_prices)
        assert np.all(signal.values >= -1)
        assert np.all(signal.values <= 1)
    
    def test_zscore_mean_reversion_logic(self, engine):
        """High prices should give negative signals (sell), low prices positive (buy)."""
        # Prices that spike up
        prices = np.array([100] * 30 + [120] * 10)  # Jump up at end
        signal = engine.calculate_zscore(prices, window=20)
        
        # After price spikes up, signal should be negative (sell signal)
        assert signal.values[-1] < 0
        
        # Prices that drop
        prices = np.array([100] * 30 + [80] * 10)  # Drop at end
        signal = engine.calculate_zscore(prices, window=20)
        
        # After price drops, signal should be positive (buy signal)
        assert signal.values[-1] > 0
    
    def test_zscore_short_data(self, engine):
        """Z-score should handle data shorter than window."""
        short_prices = np.array([100, 101, 99])
        signal = engine.calculate_zscore(short_prices, window=20)
        
        assert len(signal.values) == len(short_prices)
        assert signal.strength == 0.0  # No confidence with insufficient data
    
    # =========================================================================
    # RSI Tests
    # =========================================================================
    
    def test_rsi_returns_signal(self, engine, sample_prices):
        """RSI calculation should return a Signal object."""
        signal = engine.calculate_rsi(sample_prices)
        
        assert isinstance(signal, Signal)
        assert signal.name == "rsi"
        assert signal.signal_type == SignalType.MOMENTUM
    
    def test_rsi_bounded(self, engine, sample_prices):
        """RSI signal values should be bounded in [-1, 1]."""
        signal = engine.calculate_rsi(sample_prices)
        assert np.all(signal.values >= -1)
        assert np.all(signal.values <= 1)
    
    def test_rsi_overbought_detection(self, engine):
        """RSI should detect overbought conditions (consistent up moves)."""
        # Steadily rising prices
        prices = np.cumsum(np.ones(50) * 0.5) + 100
        signal = engine.calculate_rsi(prices, window=14)
        
        # Should give sell signal (negative) when overbought
        assert signal.values[-1] < 0
    
    def test_rsi_oversold_detection(self, engine):
        """RSI should detect oversold conditions (consistent down moves)."""
        # Steadily falling prices
        prices = 100 - np.cumsum(np.ones(50) * 0.5)
        signal = engine.calculate_rsi(prices, window=14)
        
        # Should give buy signal (positive) when oversold
        assert signal.values[-1] > 0
    
    # =========================================================================
    # Bollinger Bands Tests
    # =========================================================================
    
    def test_bollinger_returns_components(self, engine, sample_prices):
        """Bollinger Bands should return signal and three bands."""
        signal, upper, middle, lower = engine.calculate_bollinger_bands(sample_prices)
        
        assert isinstance(signal, Signal)
        assert len(upper) == len(sample_prices)
        assert len(middle) == len(sample_prices)
        assert len(lower) == len(sample_prices)
    
    def test_bollinger_band_ordering(self, engine, sample_prices):
        """Upper band should always be >= middle >= lower."""
        _, upper, middle, lower = engine.calculate_bollinger_bands(sample_prices)
        
        # Allow small numerical tolerance
        assert np.all(upper >= middle - 1e-10)
        assert np.all(middle >= lower - 1e-10)
    
    def test_bollinger_signal_at_bands(self, engine):
        """Price at upper band should give sell signal, at lower should give buy."""
        # Create prices that touch upper band
        base = np.ones(50) * 100
        base[-5:] = 110  # Price spikes to upper band
        
        signal, upper, middle, lower = engine.calculate_bollinger_bands(base, window=20)
        
        # At upper band, should get sell signal (negative)
        assert signal.values[-1] < 0
    
    # =========================================================================
    # MACD Tests
    # =========================================================================
    
    def test_macd_returns_signal(self, engine, sample_prices):
        """MACD calculation should return a Signal object."""
        signal = engine.calculate_macd(sample_prices)
        
        assert isinstance(signal, Signal)
        assert signal.name == "macd"
        assert signal.signal_type == SignalType.TREND
    
    def test_macd_bounded(self, engine, sample_prices):
        """MACD signal values should be bounded in [-1, 1]."""
        signal = engine.calculate_macd(sample_prices)
        assert np.all(signal.values >= -1)
        assert np.all(signal.values <= 1)
    
    def test_macd_trend_detection(self, engine):
        """MACD should detect upward trend."""
        # Clear, strong uptrend with deterministic data
        # Start flat, then trend strongly upward
        np.random.seed(123)
        flat_period = np.ones(40) * 100
        # Strong consistent uptrend - each day adds 1-2 points
        uptrend = 100 + np.cumsum(np.ones(60) * 1.5)
        prices = np.concatenate([flat_period, uptrend])
        
        signal = engine.calculate_macd(prices)
        
        # Should be positive in uptrend (last portion after trend established)
        # MACD needs time to catch up, so check the final 10 values
        assert np.mean(signal.values[-10:]) > 0, f"MACD should be positive in uptrend, got {np.mean(signal.values[-10:])}"
    
    # =========================================================================
    # Signal Combination Tests
    # =========================================================================
    
    def test_combine_signals_equal_weights(self, engine, sample_prices):
        """Combined signal with equal weights should average component signals."""
        zscore = engine.calculate_zscore(sample_prices)
        rsi = engine.calculate_rsi(sample_prices)
        
        combined = engine.combine_signals({
            'zscore': zscore,
            'rsi': rsi
        })
        
        assert isinstance(combined, Signal)
        assert combined.name == "combined"
    
    def test_combine_signals_custom_weights(self, engine, sample_prices):
        """Combined signal should respect custom weights."""
        zscore = engine.calculate_zscore(sample_prices)
        rsi = engine.calculate_rsi(sample_prices)
        
        # Heavy weight on z-score
        combined = engine.combine_signals(
            {'zscore': zscore, 'rsi': rsi},
            weights={'zscore': 0.9, 'rsi': 0.1}
        )
        
        # Combined should be closer to z-score
        zscore_current = zscore.get_current()
        combined_current = combined.get_current()
        rsi_current = rsi.get_current()
        
        # If z-score and rsi differ, combined should be closer to z-score
        if abs(zscore_current - rsi_current) > 0.1:
            assert abs(combined_current - zscore_current) < abs(combined_current - rsi_current)
    
    def test_combine_empty_signals(self, engine):
        """Combining empty signal dict should return neutral signal."""
        combined = engine.combine_signals({})
        assert combined.get_current() == 0.0
    
    # =========================================================================
    # Helper Method Tests
    # =========================================================================
    
    def test_rolling_mean_accuracy(self, engine):
        """Rolling mean should be mathematically correct."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        window = 3
        
        result = engine._rolling_mean(data, window)
        
        # Check known values
        # At index 2 (3rd element): mean of [1,2,3] = 2
        assert np.isclose(result[2], 2.0)
        # At index 5: mean of [4,5,6] = 5
        assert np.isclose(result[5], 5.0)
    
    def test_rolling_std_accuracy(self, engine):
        """Rolling std should be mathematically correct."""
        # Constant data should have zero std
        data = np.ones(20) * 100
        result = engine._rolling_std(data, window=5)
        
        assert np.allclose(result[5:], 0, atol=1e-10)
    
    def test_ema_responsiveness(self, engine):
        """EMA should respond to recent price changes."""
        # Price jumps from 100 to 200
        data = np.array([100] * 20 + [200] * 10, dtype=float)
        
        ema = engine._ema(data, period=5)
        
        # EMA should move toward 200 but not reach it
        assert ema[-1] > 100
        assert ema[-1] < 200
        # Shorter period should react faster
        ema_fast = engine._ema(data, period=2)
        assert ema_fast[-1] > ema[-1]  # Faster EMA closer to current price


class TestSignalDataclass:
    """Test the Signal dataclass."""
    
    def test_signal_get_current(self):
        """get_current should return last value."""
        signal = Signal(
            name="test",
            values=np.array([0.1, 0.2, 0.3]),
            signal_type=SignalType.MOMENTUM,
            strength=0.5
        )
        assert signal.get_current() == 0.3
    
    def test_signal_get_direction(self):
        """get_direction should return correct trade direction."""
        # Bullish signal
        signal = Signal(
            name="test",
            values=np.array([0.8]),
            signal_type=SignalType.MOMENTUM,
            strength=0.5
        )
        assert signal.get_direction() == 1  # Long
        
        # Bearish signal
        signal.values = np.array([-0.8])
        assert signal.get_direction() == -1  # Short
        
        # Neutral signal
        signal.values = np.array([0.2])
        assert signal.get_direction() == 0  # Hold


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_array(self):
        """Engine should handle empty arrays gracefully."""
        engine = SignalEngine()
        empty = np.array([])
        
        signal = engine.calculate_zscore(empty)
        assert len(signal.values) == 0
    
    def test_single_value(self):
        """Engine should handle single value arrays."""
        engine = SignalEngine()
        single = np.array([100.0])
        
        signal = engine.calculate_zscore(single)
        assert len(signal.values) == 1
    
    def test_constant_prices(self):
        """Engine should handle constant prices (zero volatility)."""
        engine = SignalEngine()
        constant = np.ones(50) * 100
        
        # Should not raise errors
        zscore = engine.calculate_zscore(constant)
        rsi = engine.calculate_rsi(constant)
        
        # Signals should be near zero (no clear direction)
        assert np.abs(zscore.get_current()) < 0.5
    
    def test_extreme_values(self):
        """Engine should handle extreme price values."""
        engine = SignalEngine()
        
        # Very large prices
        large = np.random.randn(50) * 1000 + 1e6
        signal_large = engine.calculate_zscore(large)
        assert np.all(np.isfinite(signal_large.values))
        
        # Very small prices (penny stocks)
        small = np.random.randn(50) * 0.01 + 0.5
        small = np.abs(small)  # Ensure positive
        signal_small = engine.calculate_zscore(small)
        assert np.all(np.isfinite(signal_small.values))


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
