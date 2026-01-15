"""
Asynchronous Market Data Handler

This module provides non-blocking concurrent data fetching for multiple
market symbols. It demonstrates async/await patterns essential for
real-time trading systems.

Key Concepts:
    - asyncio for concurrent I/O operations
    - Proper error handling for network requests
    - Data validation and normalization
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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
        return pd.DataFrame({
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }).set_index('timestamp')
    
    def __len__(self) -> int:
        return len(self.close)


class MarketDataHandler:
    """
    Asynchronous market data handler for concurrent data fetching.
    
    This class demonstrates:
        1. Async/await patterns for non-blocking I/O
        2. Concurrent task execution with asyncio.gather
        3. Data validation and error handling
        4. Synthetic data generation for testing
    
    In production, this would connect to real market data APIs
    (e.g., Interactive Brokers, Bloomberg, or crypto exchanges).
    
    Example:
        >>> handler = MarketDataHandler()
        >>> async def fetch():
        ...     data = await handler.stream_market_data(['AAPL', 'GOOGL'])
        ...     return data
        >>> results = asyncio.run(fetch())
    """
    
    def __init__(
        self,
        base_volatility: float = 0.02,
        base_drift: float = 0.0001,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the data handler.
        
        Args:
            base_volatility: Daily volatility for synthetic data generation
            base_drift: Daily drift (expected return) for synthetic data
            random_seed: Seed for reproducible synthetic data
        """
        self.base_volatility = base_volatility
        self.base_drift = base_drift
        self._cache: Dict[str, MarketData] = {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    async def stream_market_data(
        self,
        symbols: List[str],
        days: int = 252,
        use_cache: bool = True
    ) -> Dict[str, MarketData]:
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
        tasks = [
            self._fetch_symbol_data(symbol, days, use_cache)
            for symbol in symbols
        ]
        
        # Execute all tasks concurrently
        # This is the key async pattern - no blocking!
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any errors
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                print(f"Warning: Failed to fetch {symbol}: {result}")
                continue
            data_dict[symbol] = result
        
        return data_dict
    
    async def _fetch_symbol_data(
        self,
        symbol: str,
        days: int,
        use_cache: bool
    ) -> MarketData:
        """
        Fetch data for a single symbol.
        
        In production, this would make an API call to a market data provider.
        For demonstration, we generate realistic synthetic data using
        Geometric Brownian Motion (GBM).
        
        Args:
            symbol: Trading symbol
            days: Number of days of history
            use_cache: Whether to check cache first
        
        Returns:
            MarketData object with OHLCV data
        """
        cache_key = f"{symbol}_{days}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Simulate network latency (would be real API call in production)
        await asyncio.sleep(0.01)
        
        # Generate synthetic data using Geometric Brownian Motion
        data = self._generate_synthetic_data(symbol, days)
        
        if use_cache:
            self._cache[cache_key] = data
        
        return data
    
    def _generate_synthetic_data(
        self,
        symbol: str,
        days: int
    ) -> MarketData:
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
        timestamps = pd.bdate_range(
            end=end_date,
            periods=days
        ).to_pydatetime()
        
        # Starting price varies by symbol
        base_price = 100 + (symbol_seed % 400)
        
        # Generate daily returns using GBM
        # r_t = μ + σ * ε, where ε ~ N(0, 1)
        daily_returns = (
            self.base_drift + 
            self.base_volatility * rng.randn(days)
        )
        
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
        open_prices = np.roll(close_prices, 1) * (1 + rng.randn(days) * intraday_vol * 0.3)
        open_prices[0] = base_price
        
        # Ensure OHLC consistency: Low <= Open,Close <= High
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        # Generate volume (log-normal distribution is realistic)
        base_volume = 1_000_000 + (symbol_seed % 9_000_000)
        volume = rng.lognormal(
            mean=np.log(base_volume),
            sigma=0.5,
            size=days
        ).astype(int)
        
        return MarketData(
            symbol=symbol,
            timestamp=np.array(timestamps),
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            volume=volume
        )
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
    
    async def get_latest_prices(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Get the latest closing prices for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch
        
        Returns:
            Dictionary mapping symbols to their latest closing price
        """
        data = await self.stream_market_data(symbols, days=1)
        return {
            symbol: market_data.close[-1]
            for symbol, market_data in data.items()
        }


# Example usage demonstrating async patterns
async def example_usage():
    """Demonstrate async data fetching."""
    handler = MarketDataHandler(random_seed=42)
    
    # Fetch data for multiple symbols concurrently
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    print(f"Fetching data for {len(symbols)} symbols concurrently...")
    start_time = asyncio.get_event_loop().time()
    
    data = await handler.stream_market_data(symbols, days=252)
    
    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"Fetched all data in {elapsed:.3f} seconds")
    
    # Display summary
    for symbol, market_data in data.items():
        returns = np.diff(np.log(market_data.close))
        annual_vol = np.std(returns) * np.sqrt(252)
        total_return = (market_data.close[-1] / market_data.close[0]) - 1
        
        print(f"\n{symbol}:")
        print(f"  Days: {len(market_data)}")
        print(f"  Price Range: ${market_data.low.min():.2f} - ${market_data.high.max():.2f}")
        print(f"  Annual Volatility: {annual_vol:.1%}")
        print(f"  Total Return: {total_return:.1%}")


if __name__ == "__main__":
    asyncio.run(example_usage())
