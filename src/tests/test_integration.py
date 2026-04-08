"""Integration tests — verify components work together end-to-end."""

import asyncio

import numpy as np
import pytest

from src.backtest import BacktestConfig, Backtester
from src.data_handler import DataSource, MarketDataHandler
from src.risk_manager import Portfolio, RiskManager
from src.signals import SignalEngine


class TestDataHandler:
    def test_synthetic_data_fetching(self):
        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return await handler.stream_market_data(["AAPL", "GOOGL"], days=100)

        data = asyncio.run(fetch())

        assert len(data) == 2
        assert "AAPL" in data
        assert "GOOGL" in data
        assert len(data["AAPL"].close) == 100

    def test_data_has_all_ohlcv_fields(self):
        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return (await handler.stream_market_data(["AAPL"], days=50))["AAPL"]

        market_data = asyncio.run(fetch())

        assert len(market_data.open) == 50
        assert len(market_data.high) == 50
        assert len(market_data.low) == 50
        assert len(market_data.close) == 50
        assert len(market_data.volume) == 50

    def test_ohlc_consistency(self):
        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return (await handler.stream_market_data(["AAPL"], days=100))["AAPL"]

        market_data = asyncio.run(fetch())
        assert np.all(market_data.high >= market_data.low)


class TestFullPipeline:
    @pytest.fixture
    def sample_data(self):
        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return await handler.stream_market_data(["AAPL"], days=100)

        return asyncio.run(fetch())

    def test_signals_from_fetched_data(self, sample_data):
        engine = SignalEngine()
        prices = sample_data["AAPL"].close

        zscore = engine.calculate_zscore(prices)
        rsi = engine.calculate_rsi(prices)
        macd = engine.calculate_macd(prices)

        assert len(zscore.values) == len(prices)
        assert len(rsi.values) == len(prices)
        assert len(macd.values) == len(prices)

        assert np.all(np.abs(zscore.values) <= 1)
        assert np.all(np.abs(rsi.values) <= 1)
        assert np.all(np.abs(macd.values) <= 1)

    def test_risk_manager_with_real_volatility(self, sample_data):
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
        engine = SignalEngine()
        prices = sample_data["AAPL"].close

        zscore = engine.calculate_zscore(prices)
        signals = zscore.values.reshape(-1, 1)
        prices_bt = prices.reshape(-1, 1)

        config = BacktestConfig(initial_capital=100000)
        backtester = Backtester(config)
        metrics, equity, positions = backtester.run(prices_bt, signals)

        assert len(equity) == len(prices)
        assert equity[0] == 100000
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown >= 0

    def test_full_pipeline_end_to_end(self):
        """Fetch data -> generate signals -> build matrices -> backtest."""

        async def fetch():
            handler = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)
            return await handler.stream_market_data(["AAPL", "MSFT"], days=100)

        data = asyncio.run(fetch())
        engine = SignalEngine()

        min_len = min(len(d.close) for d in data.values())
        prices = np.column_stack([d.close[:min_len] for d in data.values()])
        signals = np.column_stack(
            [
                engine.combine_signals(
                    {
                        "zscore": engine.calculate_zscore(d.close[:min_len]),
                        "rsi": engine.calculate_rsi(d.close[:min_len]),
                    }
                ).values
                for d in data.values()
            ]
        )

        backtester = Backtester(BacktestConfig(initial_capital=100000))
        metrics, equity, _ = backtester.run(prices, signals)

        assert len(equity) > 0
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
