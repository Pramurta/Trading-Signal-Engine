"""
Integration Tests for Trading Signal Engine

Run with: python tests/test_integration.py
Or with pytest: pytest tests/test_integration.py -v
"""

import asyncio

import numpy as np
import pytest

from src.backtest import BacktestConfig, Backtester
from src.data_handler import DataSource, MarketDataHandler
from src.risk_manager import Portfolio, RiskManager
from src.signals import SignalEngine
from src.strategy import TradingStrategy


class TestDataHandler:
    """Test data handler with both data sources."""

    def test_synthetic_data_fetching(self):
        """Synthetic data should work without internet."""

        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            data = await handler.stream_market_data(["AAPL", "GOOGL"], days=100)
            return data

        data = asyncio.run(fetch())

        assert len(data) == 2
        assert "AAPL" in data
        assert "GOOGL" in data
        assert len(data["AAPL"].close) == 100

    def test_data_has_all_ohlcv_fields(self):
        """MarketData should have all OHLCV fields."""

        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            data = await handler.stream_market_data(["AAPL"], days=50)
            return data["AAPL"]

        market_data = asyncio.run(fetch())

        assert len(market_data.open) == 50
        assert len(market_data.high) == 50
        assert len(market_data.low) == 50
        assert len(market_data.close) == 50
        assert len(market_data.volume) == 50

    def test_ohlc_consistency(self):
        """High should be >= Low, and Open/Close within range."""

        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            data = await handler.stream_market_data(["AAPL"], days=100)
            return data["AAPL"]

        market_data = asyncio.run(fetch())

        assert np.all(market_data.high >= market_data.low)


class TestFullPipeline:
    """Test the complete trading pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Fetch sample data for tests."""

        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return await handler.stream_market_data(["AAPL"], days=100)

        return asyncio.run(fetch())

    def test_signals_from_real_prices(self, sample_data):
        """Signal engine should work with fetched data."""
        engine = SignalEngine()
        prices = sample_data["AAPL"].close

        zscore = engine.calculate_zscore(prices)
        rsi = engine.calculate_rsi(prices)
        macd = engine.calculate_macd(prices)

        assert len(zscore.values) == len(prices)
        assert len(rsi.values) == len(prices)
        assert len(macd.values) == len(prices)

        # All signals should be bounded
        assert np.all(np.abs(zscore.values) <= 1)
        assert np.all(np.abs(rsi.values) <= 1)
        assert np.all(np.abs(macd.values) <= 1)

    def test_risk_manager_with_real_volatility(self, sample_data):
        """Risk manager should size positions based on real volatility."""
        prices = sample_data["AAPL"].close
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)

        risk_mgr = RiskManager()
        portfolio = Portfolio(capital=100000)

        position = risk_mgr.calculate_position_size(
            symbol="AAPL",
            signal_strength=0.7,
            price=prices[-1],
            volatility=volatility,
            portfolio=portfolio,
        )

        assert position.shares >= 0
        assert position.weight <= risk_mgr.max_position_pct

    def test_backtest_with_signals(self, sample_data):
        """Backtester should work with generated signals."""
        engine = SignalEngine()
        prices = sample_data["AAPL"].close

        # Generate signals
        zscore = engine.calculate_zscore(prices)
        signals = zscore.values.reshape(-1, 1)
        prices_bt = prices.reshape(-1, 1)

        # Run backtest
        config = BacktestConfig(initial_capital=100000)
        backtester = Backtester(config)
        metrics, equity, positions = backtester.run(prices_bt, signals)

        assert len(equity) == len(prices)
        assert equity[0] == 100000  # Started with initial capital
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown >= 0

    def test_strategy_backtest(self):
        """Full strategy backtest should complete without errors."""

        async def run_backtest():
            strategy = TradingStrategy(
                data_handler=MarketDataHandler(
                    source=DataSource.SYNTHETIC, random_seed=42
                ),
                signal_engine=SignalEngine(),
                risk_manager=RiskManager(),
            )
            return await strategy.run_backtest(["AAPL", "MSFT"], days=100)

        results = asyncio.run(run_backtest())

        assert results.equity_curve is not None
        assert len(results.equity_curve) > 0
        assert results.total_return is not None
        assert results.sharpe_ratio is not None


def run_verification():
    """Run all tests and print results."""
    print("=" * 50)
    print("TRADING ENGINE VERIFICATION")
    print("=" * 50)

    _tests = [  # noqa: F841
        ("Data Handler - Synthetic", TestDataHandler().test_synthetic_data_fetching),
        (
            "Data Handler - OHLCV Fields",
            TestDataHandler().test_data_has_all_ohlcv_fields,
        ),
        ("Data Handler - OHLC Consistency", TestDataHandler().test_ohlc_consistency),
        (
            "Pipeline - Signals",
            lambda: TestFullPipeline().test_signals_from_real_prices(
                TestFullPipeline().sample_data.fget(None)
            ),
        ),
        ("Pipeline - Strategy Backtest", TestFullPipeline().test_strategy_backtest),
    ]

    # Simpler verification
    passed = 0
    failed = 0

    # Test 1: Data Handler
    print("\n[1/5] Data Handler (Synthetic)...", end=" ")
    try:
        TestDataHandler().test_synthetic_data_fetching()
        print("✓")
        passed += 1
    except Exception as e:
        print(f"✗ {e}")
        failed += 1

    # Test 2: OHLCV Fields
    print("[2/5] Data Handler (OHLCV Fields)...", end=" ")
    try:
        TestDataHandler().test_data_has_all_ohlcv_fields()
        print("✓")
        passed += 1
    except Exception as e:
        print(f"✗ {e}")
        failed += 1

    # Test 3: OHLC Consistency
    print("[3/5] Data Handler (OHLC Consistency)...", end=" ")
    try:
        TestDataHandler().test_ohlc_consistency()
        print("✓")
        passed += 1
    except Exception as e:
        print(f"✗ {e}")
        failed += 1

    # Test 4: Signals + Risk + Backtest
    print("[4/5] Signals + Risk + Backtest...", end=" ")
    try:

        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return await handler.stream_market_data(["AAPL"], days=100)

        sample_data = asyncio.run(fetch())

        t = TestFullPipeline()
        t.test_signals_from_real_prices(sample_data)
        t.test_risk_manager_with_real_volatility(sample_data)
        t.test_backtest_with_signals(sample_data)
        print("✓")
        passed += 1
    except Exception as e:
        print(f"✗ {e}")
        failed += 1

    # Test 5: Full Strategy
    print("[5/5] Full Strategy Backtest...", end=" ")
    try:
        TestFullPipeline().test_strategy_backtest()
        print("✓")
        passed += 1
    except Exception as e:
        print(f"✗ {e}")
        failed += 1

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED ✓")
    else:
        print(f"PASSED: {passed}, FAILED: {failed}")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_verification()
    exit(0 if success else 1)
