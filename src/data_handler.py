"""
Asynchronous Market Data Handler

This module provides non-blocking concurrent data fetching for multiple
market symbols. It demonstrates async/await patterns essential for
real-time trading systems.

Key Concepts:
    - asyncio for concurrent I/O operations
    - Proper error handling for network requests
    - Data validation and normalization
    - Real market data via yfinance
    - Synthetic data generation for testing
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

# Try to import yfinance, but don't fail if not installed
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class DataSource(Enum):
    """Available data sources."""

    SYNTHETIC = "synthetic"  # Generated fake data (no internet required)
    YFINANCE = "yfinance"  # Real data from Yahoo Finance


@dataclass
class MarketData:
    """
    Container for OHLCV market data.

    Attributes:
        symbol: Trading symbol (e.g., 'AAPL')
        timestamp: Array of datetime objects
        open: Opening prices
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Trading volume
    """

    symbol: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
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
    """
    Asynchronous market data handler for concurrent data fetching.

    This class demonstrates:
        1. Async/await patterns for non-blocking I/O
        2. Concurrent task execution with asyncio.gather
        3. Data validation and error handling
        4. Real data fetching via yfinance
        5. Synthetic data generation for testing

    Data Sources:
        - SYNTHETIC: Generated data using Geometric Brownian Motion (no internet)
        - YFINANCE: Real market data from Yahoo Finance (requires internet)

    Example:
        >>> # Using real data
        >>> handler = MarketDataHandler(source=DataSource.YFINANCE)
        >>> async def fetch():
        ...     data = await handler.stream_market_data(['AAPL', 'GOOGL'])
        ...     return data
        >>> results = asyncio.run(fetch())

        >>> # Using synthetic data (offline/testing)
        >>> handler = MarketDataHandler(source=DataSource.SYNTHETIC)
        >>> results = asyncio.run(handler.stream_market_data(['AAPL']))
    """

    def __init__(
        self,
        source: DataSource = DataSource.SYNTHETIC,
        base_volatility: float = 0.02,
        base_drift: float = 0.0001,
        random_seed: int | None = None,
    ):
        """
        Initialize the data handler.

        Args:
            source: Data source - SYNTHETIC or YFINANCE
            base_volatility: Daily volatility for synthetic data generation
            base_drift: Daily drift (expected return) for synthetic data
            random_seed: Seed for reproducible synthetic data

        Raises:
            ImportError: If YFINANCE source is selected but yfinance not installed
        """
        self.source = source
        self.base_volatility = base_volatility
        self.base_drift = base_drift
        self._cache: dict[str, MarketData] = {}

        # Validate yfinance availability
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
        """
        Fetch market data for multiple symbols concurrently.

        This method demonstrates async programming:
            - Creates tasks for each symbol
            - Executes all tasks concurrently with gather()
            - Returns when all data is fetched

        Args:
            symbols: List of trading symbols to fetch
            days: Number of historical days to retrieve
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping symbols to MarketData objects

        Example:
            >>> handler = MarketDataHandler()
            >>> data = await handler.stream_market_data(['AAPL', 'MSFT'])
            >>> print(data['AAPL'].close[-1])  # Latest close price
        """
        # Create concurrent tasks for each symbol
        tasks = [self._fetch_symbol_data(symbol, days, use_cache) for symbol in symbols]

        # Execute all tasks concurrently
        # This is the key async pattern - no blocking!
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle any errors
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
        """
        Fetch data for a single symbol.

        Routes to appropriate data source (yfinance or synthetic).

        Args:
            symbol: Trading symbol
            days: Number of days of history
            use_cache: Whether to check cache first

        Returns:
            MarketData object with OHLCV data
        """
        cache_key = f"{self.source.value}_{symbol}_{days}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch based on data source
        if self.source == DataSource.YFINANCE:
            data = await self._fetch_yfinance_data(symbol, days)
        else:
            # Simulate network latency for synthetic data
            await asyncio.sleep(0.01)
            data = self._generate_synthetic_data(symbol, days)

        if use_cache:
            self._cache[cache_key] = data

        return data

    async def _fetch_yfinance_data(self, symbol: str, days: int) -> MarketData:
        """
        Fetch real market data from Yahoo Finance.

        Uses a thread pool executor to run yfinance (which is blocking)
        in a non-blocking way.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
            days: Number of trading days of history

        Returns:
            MarketData object with real OHLCV data

        Raises:
            ValueError: If no data is returned for the symbol
        """
        # yfinance is blocking, so run in thread pool to keep async
        loop = asyncio.get_event_loop()

        def fetch_blocking():
            # Calculate period needed (add buffer for weekends/holidays)
            calendar_days = int(days * 1.5) + 10

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{calendar_days}d")

            if df.empty:
                raise ValueError(f"No data returned for symbol: {symbol}")

            # Limit to requested number of trading days
            df = df.tail(days)

            return df

        # Run blocking call in thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(executor, fetch_blocking)

        # Convert to MarketData
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
        """
        Generate realistic synthetic OHLCV data using Geometric Brownian Motion.

        GBM Model:
            dS = μSdt + σSdW

            where:
                S = stock price
                μ = drift (expected return)
                σ = volatility
                W = Wiener process (random walk)

        This is the standard model used in quantitative finance for
        simulating asset prices.

        Args:
            symbol: Symbol name (used to seed consistent randomness)
            days: Number of trading days to generate

        Returns:
            MarketData object with synthetic OHLCV data
        """
        # Use symbol hash for reproducible but varied data per symbol
        symbol_seed = hash(symbol) % (2**31)
        rng = np.random.RandomState(symbol_seed)

        # Generate timestamps (business days)
        end_date = datetime.now()
        timestamps = pd.bdate_range(end=end_date, periods=days).to_pydatetime()

        # Starting price varies by symbol
        base_price = 100 + (symbol_seed % 400)

        # Generate daily returns using GBM
        # r_t = μ + σ * ε, where ε ~ N(0, 1)
        daily_returns = self.base_drift + self.base_volatility * rng.randn(days)

        # Convert returns to prices: P_t = P_0 * exp(Σr_i)
        cumulative_returns = np.cumsum(daily_returns)
        close_prices = base_price * np.exp(cumulative_returns)

        # Generate realistic OHLC from close prices
        # Intraday volatility is typically 60-80% of daily volatility
        intraday_vol = self.base_volatility * 0.7

        # High is close + positive deviation
        high_prices = close_prices * (1 + np.abs(rng.randn(days)) * intraday_vol)

        # Low is close - positive deviation
        low_prices = close_prices * (1 - np.abs(rng.randn(days)) * intraday_vol)

        # Open is previous close + overnight gap
        open_prices = np.roll(close_prices, 1) * (
            1 + rng.randn(days) * intraday_vol * 0.3
        )
        open_prices[0] = base_price

        # Ensure OHLC consistency: Low <= Open,Close <= High
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

        # Generate volume (log-normal distribution is realistic)
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
        """
        Get the latest closing prices for multiple symbols.

        Args:
            symbols: List of symbols to fetch

        Returns:
            Dictionary mapping symbols to their latest closing price
        """
        data = await self.stream_market_data(symbols, days=1)
        return {symbol: market_data.close[-1] for symbol, market_data in data.items()}


# Example usage demonstrating async patterns
async def example_usage():
    """Demonstrate async data fetching with both data sources."""

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    # =========================================================================
    # Example 1: Synthetic Data (no internet required)
    # =========================================================================
    print("=" * 60)
    print("Example 1: SYNTHETIC DATA")
    print("=" * 60)

    handler_synthetic = MarketDataHandler(source=DataSource.SYNTHETIC, random_seed=42)

    print(f"\nFetching synthetic data for {len(symbols)} symbols...")
    data = await handler_synthetic.stream_market_data(symbols, days=252)

    for symbol, market_data in data.items():
        returns = np.diff(np.log(market_data.close))
        annual_vol = np.std(returns) * np.sqrt(252)
        total_return = (market_data.close[-1] / market_data.close[0]) - 1

        print(
            f"  {symbol}: ${market_data.close[-1]:.2f} | "
            f"Vol: {annual_vol:.1%} | Return: {total_return:+.1%}"
        )

    # =========================================================================
    # Example 2: Real Data from Yahoo Finance
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: REAL DATA (Yahoo Finance)")
    print("=" * 60)

    if YFINANCE_AVAILABLE:
        try:
            handler_real = MarketDataHandler(source=DataSource.YFINANCE)

            print(f"\nFetching real data for {len(symbols)} symbols...")
            real_data = await handler_real.stream_market_data(symbols, days=60)

            for symbol, market_data in real_data.items():
                returns = np.diff(np.log(market_data.close))
                annual_vol = np.std(returns) * np.sqrt(252)
                total_return = (market_data.close[-1] / market_data.close[0]) - 1

                print(
                    f"  {symbol}: ${market_data.close[-1]:.2f} | "
                    f"Vol: {annual_vol:.1%} | Return: {total_return:+.1%}"
                )

        except Exception as e:
            print(f"\n  Could not fetch real data: {e}")
            print("  (This is expected if you don't have internet access)")
    else:
        print("\n  yfinance not installed. Install with: pip install yfinance")


async def quick_demo_real_data():
    """
    Quick demo to fetch real data for a single stock.

    Usage:
        >>> import asyncio
        >>> from src.data_handler import quick_demo_real_data
        >>> asyncio.run(quick_demo_real_data())
    """
    if not YFINANCE_AVAILABLE:
        print("Install yfinance first: pip install yfinance")
        return

    handler = MarketDataHandler(source=DataSource.YFINANCE)
    data = await handler.stream_market_data(["AAPL"], days=30)

    aapl = data["AAPL"]
    df = aapl.to_dataframe()

    print("\nAAPL - Last 30 Trading Days")
    print("-" * 40)
    print(f"Latest Close: ${aapl.close[-1]:.2f}")
    print(f"30-Day High:  ${aapl.high.max():.2f}")
    print(f"30-Day Low:   ${aapl.low.min():.2f}")
    print(f"Avg Volume:   {aapl.volume.mean():,.0f}")

    return df


if __name__ == "__main__":
    asyncio.run(example_usage())
