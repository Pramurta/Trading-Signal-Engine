"""Async market data handler — yfinance or synthetic OHLCV."""

import asyncio
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class DataSource(Enum):
    SYNTHETIC = "synthetic"
    YFINANCE = "yfinance"


@dataclass
class MarketData:
    """Container for OHLCV market data."""

    symbol: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame indexed by timestamp."""
        return pd.DataFrame(
            {
                "timestamp": self.timestamp,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            }
        ).set_index("timestamp")

    def __len__(self) -> int:
        return len(self.close)


class MarketDataHandler:
    """Fetches market data for multiple symbols concurrently using asyncio."""

    def __init__(
        self,
        source: DataSource = DataSource.SYNTHETIC,
        base_volatility: float = 0.02,
        base_drift: float = 0.0001,
        random_seed: int | None = None,
    ):
        self.source = source
        self.base_volatility = base_volatility
        self.base_drift = base_drift
        self._cache: dict[str, MarketData] = {}

        if source == DataSource.YFINANCE and not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required for real market data. "
                "Install with: pip install yfinance"
            )

        if random_seed is not None:
            np.random.seed(random_seed)

    async def stream_market_data(
        self, symbols: list[str], days: int = 252, use_cache: bool = True
    ) -> dict[str, MarketData]:
        """Fetch data for multiple symbols concurrently."""
        tasks = [self._fetch_symbol_data(symbol, days, use_cache) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_dict: dict[str, MarketData] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, BaseException):
                print(f"Warning: Failed to fetch {symbol}: {result}")
                continue
            data_dict[symbol] = result

        return data_dict

    async def _fetch_symbol_data(
        self, symbol: str, days: int, use_cache: bool
    ) -> MarketData:
        """Fetch data for a single symbol, routing to yfinance or synthetic."""
        cache_key = f"{self.source.value}_{symbol}_{days}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self.source == DataSource.YFINANCE:
            data = await self._fetch_yfinance_data(symbol, days)
        else:
            await asyncio.sleep(0.01)
            data = self._generate_synthetic_data(symbol, days)

        if use_cache:
            self._cache[cache_key] = data

        return data

    async def _fetch_yfinance_data(self, symbol: str, days: int) -> MarketData:
        """Fetch real data from Yahoo Finance via thread pool (yfinance is blocking)."""
        loop = asyncio.get_event_loop()

        def fetch_blocking():
            calendar_days = int(days * 1.5) + 10
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{calendar_days}d")

            if df.empty:
                raise ValueError(f"No data returned for symbol: {symbol}")

            return df.tail(days)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(executor, fetch_blocking)

        return MarketData(
            symbol=symbol,
            timestamp=df.index.to_pydatetime(),
            open=df["Open"].values,
            high=df["High"].values,
            low=df["Low"].values,
            close=df["Close"].values,
            volume=df["Volume"].values.astype(int),
        )

    def _generate_synthetic_data(self, symbol: str, days: int) -> MarketData:
        """Generate synthetic OHLCV data using Geometric Brownian Motion."""
        symbol_seed = hash(symbol) % (2**31)
        rng = np.random.RandomState(symbol_seed)

        end_date = datetime.now()
        timestamps = pd.bdate_range(end=end_date, periods=days).to_pydatetime()

        base_price = 100 + (symbol_seed % 400)

        # GBM: r_t = drift + vol * epsilon
        daily_returns = self.base_drift + self.base_volatility * rng.randn(days)
        cumulative_returns = np.cumsum(daily_returns)
        close_prices = base_price * np.exp(cumulative_returns)

        # Intraday OHLC from close prices
        intraday_vol = self.base_volatility * 0.7
        high_prices = close_prices * (1 + np.abs(rng.randn(days)) * intraday_vol)
        low_prices = close_prices * (1 - np.abs(rng.randn(days)) * intraday_vol)

        open_prices = np.roll(close_prices, 1) * (
            1 + rng.randn(days) * intraday_vol * 0.3
        )
        open_prices[0] = base_price

        # Enforce OHLC consistency
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

        base_volume = 1_000_000 + (symbol_seed % 9_000_000)
        volume = rng.lognormal(mean=np.log(base_volume), sigma=0.5, size=days).astype(
            int
        )

        return MarketData(
            symbol=symbol,
            timestamp=np.array(timestamps),
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            volume=volume,
        )

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

    async def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get the latest closing price for each symbol."""
        data = await self.stream_market_data(symbols, days=1)
        return {symbol: market_data.close[-1] for symbol, market_data in data.items()}
