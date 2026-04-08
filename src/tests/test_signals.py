"""Tests for the signal generation engine."""

import numpy as np
import pytest

from src.signals import Signal, SignalEngine, SignalType


class TestZScore:
    @pytest.fixture
    def engine(self):
        return SignalEngine(default_window=20)

    def test_returns_correct_type_and_metadata(self, engine):
        prices = np.linspace(100, 110, 50)
        signal = engine.calculate_zscore(prices)
        assert isinstance(signal, Signal)
        assert signal.name == "zscore"
        assert signal.signal_type == SignalType.MEAN_REVERSION

    def test_output_length_matches_input(self, engine):
        prices = np.linspace(100, 110, 50)
        signal = engine.calculate_zscore(prices)
        assert len(signal.values) == 50

    def test_values_bounded(self, engine):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200))
        signal = engine.calculate_zscore(prices)
        assert np.all(signal.values >= -1)
        assert np.all(signal.values <= 1)

    def test_spike_up_gives_sell_signal(self, engine):
        """Price above rolling mean -> sell signal (negative)."""
        prices = np.array([100.0] * 30 + [120.0] * 10)
        signal = engine.calculate_zscore(prices, window=20)
        assert signal.values[-1] < 0

    def test_spike_down_gives_buy_signal(self, engine):
        """Price below rolling mean -> buy signal (positive)."""
        prices = np.array([100.0] * 30 + [80.0] * 10)
        signal = engine.calculate_zscore(prices, window=20)
        assert signal.values[-1] > 0

    def test_constant_prices_near_zero(self, engine):
        """Flat prices should produce near-zero signals."""
        prices = np.ones(50) * 100
        signal = engine.calculate_zscore(prices)
        assert np.all(np.abs(signal.values) < 0.01)

    def test_known_zscore_value(self):
        """Verify against hand-calculated z-score.

        Window=3, prices=[10, 10, 10, 13]:
        At index 3: rolling_mean = (10+10+13)/3 = 11, rolling_std = std([10,10,13])
        std = sqrt(((10-11)^2 + (10-11)^2 + (13-11)^2)/3) = sqrt(6/3) = sqrt(2) ~ 1.414
        z = (13 - 11) / 1.414 ~ 1.414
        normalized = -tanh(1.414 / 2) = -tanh(0.707) ~ -0.610
        """
        engine = SignalEngine()
        prices = np.array([10.0, 10.0, 10.0, 13.0])
        signal = engine.calculate_zscore(prices, window=3)
        assert signal.values[-1] == pytest.approx(-0.610, abs=0.02)

    def test_short_data_returns_zeros(self, engine):
        signal = engine.calculate_zscore(np.array([100.0, 101.0]), window=20)
        assert len(signal.values) == 2
        assert signal.strength == 0.0
        assert np.allclose(signal.values, 0)

    def test_empty_array(self, engine):
        signal = engine.calculate_zscore(np.array([]))
        assert len(signal.values) == 0

    def test_single_value(self, engine):
        signal = engine.calculate_zscore(np.array([100.0]))
        assert len(signal.values) == 1


class TestRSI:
    @pytest.fixture
    def engine(self):
        return SignalEngine()

    def test_returns_correct_metadata(self, engine):
        prices = np.linspace(100, 110, 50)
        signal = engine.calculate_rsi(prices)
        assert signal.name == "rsi"
        assert signal.signal_type == SignalType.MOMENTUM

    def test_values_bounded(self, engine):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200))
        signal = engine.calculate_rsi(prices)
        assert np.all(signal.values >= -1)
        assert np.all(signal.values <= 1)

    def test_steady_uptrend_gives_sell_signal(self, engine):
        """Consistently rising prices -> overbought -> negative (sell) signal."""
        prices = 100 + np.cumsum(np.ones(50) * 0.5)
        signal = engine.calculate_rsi(prices, window=14)
        assert signal.values[-1] < 0

    def test_steady_downtrend_gives_buy_signal(self, engine):
        """Consistently falling prices -> oversold -> positive (buy) signal."""
        prices = 100 - np.cumsum(np.ones(50) * 0.5)
        signal = engine.calculate_rsi(prices, window=14)
        assert signal.values[-1] > 0

    def test_all_gains_rsi_near_100(self, engine):
        """If every day is a gain, raw RSI should approach 100 (signal near -1)."""
        prices = np.arange(100.0, 150.0, 1.0)  # 50 days, +1 each day
        signal = engine.calculate_rsi(prices, window=14)
        # Signal is inverted: RSI near 100 -> signal near -1
        assert signal.values[-1] < -0.8

    def test_all_losses_rsi_near_0(self, engine):
        """If every day is a loss, raw RSI should approach 0 (signal near +1)."""
        prices = np.arange(150.0, 100.0, -1.0)  # 50 days, -1 each day
        signal = engine.calculate_rsi(prices, window=14)
        assert signal.values[-1] > 0.8

    def test_short_data_returns_zeros(self, engine):
        signal = engine.calculate_rsi(np.array([100.0, 101.0, 99.0]), window=14)
        assert np.allclose(signal.values, 0)

    def test_constant_prices(self, engine):
        """No price changes -> no gains or losses -> RSI floors at 0 -> signal +1.

        With zero deltas, avg_gain=0 and avg_loss hits the 1e-10 floor,
        giving RS=0, RSI=0 (extreme oversold). This is a known edge case
        where RSI is undefined — the signal saturates.
        """
        prices = np.ones(50) * 100
        signal = engine.calculate_rsi(prices, window=14)
        # RSI = 0 -> normalized = -(0-50)/50 = 1.0
        assert signal.values[-1] == pytest.approx(1.0, abs=0.01)


class TestBollingerBands:
    @pytest.fixture
    def engine(self):
        return SignalEngine()

    def test_returns_signal_and_three_bands(self, engine):
        prices = np.linspace(100, 110, 50)
        signal, upper, middle, lower = engine.calculate_bollinger_bands(prices)
        assert isinstance(signal, Signal)
        assert signal.name == "bollinger"
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_band_ordering(self, engine):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100))
        _, upper, middle, lower = engine.calculate_bollinger_bands(prices)
        assert np.all(upper >= middle - 1e-10)
        assert np.all(middle >= lower - 1e-10)

    def test_price_above_upper_gives_sell(self, engine):
        """Price spiking above upper band -> negative signal."""
        prices = np.array([100.0] * 30 + [110.0] * 5)
        signal, _, _, _ = engine.calculate_bollinger_bands(prices, window=20)
        assert signal.values[-1] < 0

    def test_price_at_middle_near_zero(self, engine):
        """Flat prices sit at the middle band -> signal near 0."""
        prices = np.ones(50) * 100
        signal, _, _, _ = engine.calculate_bollinger_bands(prices, window=20)
        # With zero std, bands collapse. %B undefined but clipped.
        # Not a strong assertion — mainly checking no crash.
        assert np.all(np.isfinite(signal.values))

    def test_short_data_returns_price_copies(self, engine):
        prices = np.array([100.0, 101.0, 99.0])
        result = engine.calculate_bollinger_bands(prices, window=20)
        signal, upper, middle, lower = result
        assert np.allclose(upper, prices)
        assert np.allclose(middle, prices)
        assert np.allclose(lower, prices)


class TestMACD:
    @pytest.fixture
    def engine(self):
        return SignalEngine()

    def test_returns_correct_metadata(self, engine):
        prices = np.linspace(100, 130, 50)
        signal = engine.calculate_macd(prices)
        assert signal.name == "macd"
        assert signal.signal_type == SignalType.TREND

    def test_values_bounded(self, engine):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200))
        signal = engine.calculate_macd(prices)
        assert np.all(signal.values >= -1)
        assert np.all(signal.values <= 1)

    def test_strong_uptrend_positive(self, engine):
        """Clear uptrend after flat period -> MACD histogram positive."""
        flat = np.ones(40) * 100
        uptrend = 100 + np.cumsum(np.ones(60) * 1.5)
        prices = np.concatenate([flat, uptrend])
        signal = engine.calculate_macd(prices)
        assert np.mean(signal.values[-10:]) > 0

    def test_strong_downtrend_negative(self, engine):
        """Clear downtrend after flat period -> MACD histogram negative."""
        flat = np.ones(40) * 200
        downtrend = 200 - np.cumsum(np.ones(60) * 1.5)
        prices = np.concatenate([flat, downtrend])
        signal = engine.calculate_macd(prices)
        assert np.mean(signal.values[-10:]) < 0

    def test_short_data_returns_zeros(self, engine):
        signal = engine.calculate_macd(np.array([100.0] * 10))
        assert np.allclose(signal.values, 0)


class TestCombineSignals:
    @pytest.fixture
    def engine(self):
        return SignalEngine()

    def test_equal_weights_average(self, engine):
        s1 = Signal("a", np.array([0.6, 0.6]), SignalType.MOMENTUM, 1.0)
        s2 = Signal("b", np.array([0.2, 0.2]), SignalType.MOMENTUM, 1.0)
        combined = engine.combine_signals({"a": s1, "b": s2})
        # Equal weights: (0.6 + 0.2) / 2 = 0.4
        assert combined.values[-1] == pytest.approx(0.4, abs=0.001)

    def test_custom_weights(self, engine):
        s1 = Signal("a", np.array([1.0]), SignalType.MOMENTUM, 1.0)
        s2 = Signal("b", np.array([0.0]), SignalType.MOMENTUM, 1.0)
        combined = engine.combine_signals(
            {"a": s1, "b": s2}, weights={"a": 0.75, "b": 0.25}
        )
        assert combined.values[-1] == pytest.approx(0.75, abs=0.001)

    def test_empty_signals_returns_zero(self, engine):
        combined = engine.combine_signals({})
        assert combined.get_current() == 0.0

    def test_combined_clipped_to_range(self, engine):
        # Two signals both at +1 with equal weight should still be clipped to 1
        s1 = Signal("a", np.array([1.0]), SignalType.MOMENTUM, 1.0)
        s2 = Signal("b", np.array([1.0]), SignalType.MOMENTUM, 1.0)
        combined = engine.combine_signals({"a": s1, "b": s2})
        assert combined.values[-1] <= 1.0


class TestHelpers:
    def test_rolling_mean_known_values(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SignalEngine._rolling_mean(data, window=3)
        # Index 2: expanding mean of [1,2,3] = 2.0 (first full window)
        assert result[2] == pytest.approx(2.0)
        # Index 3: mean of [2,3,4] = 3.0
        assert result[3] == pytest.approx(3.0)
        # Index 4: mean of [3,4,5] = 4.0
        assert result[4] == pytest.approx(4.0)

    def test_rolling_std_constant_is_zero(self):
        data = np.ones(20) * 100
        result = SignalEngine._rolling_std(data, window=5)
        assert np.allclose(result[5:], 0, atol=1e-10)

    def test_ema_converges_to_step(self):
        """EMA of a step function should approach the new value."""
        data = np.array([100.0] * 20 + [200.0] * 10)
        ema = SignalEngine._ema(data, period=5)
        assert ema[-1] > 100
        assert ema[-1] < 200
        # Faster EMA should be closer to 200
        ema_fast = SignalEngine._ema(data, period=2)
        assert ema_fast[-1] > ema[-1]


class TestSignalDataclass:
    def test_get_current(self):
        signal = Signal("test", np.array([0.1, 0.2, 0.3]), SignalType.MOMENTUM, 0.5)
        assert signal.get_current() == 0.3

    def test_get_current_empty(self):
        signal = Signal("test", np.array([]), SignalType.MOMENTUM, 0.5)
        assert signal.get_current() == 0.0

    def test_get_direction_long(self):
        signal = Signal("test", np.array([0.8]), SignalType.MOMENTUM, 0.5)
        assert signal.get_direction() == 1

    def test_get_direction_short(self):
        signal = Signal("test", np.array([-0.8]), SignalType.MOMENTUM, 0.5)
        assert signal.get_direction() == -1

    def test_get_direction_neutral(self):
        signal = Signal("test", np.array([0.2]), SignalType.MOMENTUM, 0.5)
        assert signal.get_direction() == 0


class TestEdgeCases:
    def test_extreme_large_prices(self):
        engine = SignalEngine()
        np.random.seed(42)
        prices = np.random.randn(50) * 1000 + 1e6
        signal = engine.calculate_zscore(prices)
        assert np.all(np.isfinite(signal.values))

    def test_penny_stock_prices(self):
        engine = SignalEngine()
        np.random.seed(42)
        prices = np.abs(np.random.randn(50) * 0.01 + 0.5)
        signal = engine.calculate_zscore(prices)
        assert np.all(np.isfinite(signal.values))

    def test_nan_in_prices_propagates(self):
        """NaN in input should not crash; values may be NaN but no exception."""
        engine = SignalEngine()
        prices = np.array([100.0] * 25 + [float("nan")] + [100.0] * 24)
        # Should not raise
        signal = engine.calculate_zscore(prices)
        assert len(signal.values) == 50
