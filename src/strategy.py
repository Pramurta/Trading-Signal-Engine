"""Trading strategy orchestrator — ties together data, signals, and risk management."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .data_handler import MarketData, MarketDataHandler
from .risk_manager import Portfolio, PositionSize, RiskManager
from .signals import Signal, SignalEngine


@dataclass
class TradeDecision:
    """A trade decision with signal context and position sizing."""

    symbol: str
    direction: int  # 1 (long), -1 (short), 0 (no trade)
    position_size: PositionSize
    signals: dict[str, float]
    combined_signal: float
    timestamp: datetime
    reason: str

    def __str__(self) -> str:
        direction_str = {1: "LONG", -1: "SHORT", 0: "HOLD"}[self.direction]
        return (
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{self.symbol} | {direction_str} | "
            f"Signal: {self.combined_signal:+.2f} | "
            f"{self.position_size}"
        )


class TradingStrategy:
    """Orchestrates data, signals, and risk-managed trade decisions."""

    def __init__(
        self,
        data_handler: MarketDataHandler,
        signal_engine: SignalEngine,
        risk_manager: RiskManager,
        signal_weights: dict[str, float] | None = None,
        signal_threshold: float = 0.3,
    ):
        self.data_handler = data_handler
        self.signal_engine = signal_engine
        self.risk_manager = risk_manager

        self.signal_weights = signal_weights or {
            "zscore": 0.35,
            "rsi": 0.25,
            "macd": 0.25,
            "bollinger": 0.15,
        }

        self.signal_threshold = signal_threshold

    async def generate_signals(
        self, symbol: str, data: MarketData
    ) -> tuple[dict[str, Signal], Signal]:
        """Calculate all signals for a symbol and return individual + combined."""
        prices = data.close

        zscore = self.signal_engine.calculate_zscore(prices, window=20)
        rsi = self.signal_engine.calculate_rsi(prices, window=14)
        macd = self.signal_engine.calculate_macd(prices)
        bollinger, _, _, _ = self.signal_engine.calculate_bollinger_bands(prices)

        signals = {"zscore": zscore, "rsi": rsi, "macd": macd, "bollinger": bollinger}
        combined = self.signal_engine.combine_signals(signals, self.signal_weights)

        return signals, combined

    async def generate_trade_decision(
        self,
        symbol: str,
        data: MarketData,
        portfolio: Portfolio,
        win_rate: float = 0.52,
        win_loss_ratio: float = 1.3,
    ) -> TradeDecision:
        """Generate a trade decision with direction and position size."""
        signals, combined = await self.generate_signals(symbol, data)

        current_price = data.close[-1]
        current_signal = combined.get_current()

        returns = np.diff(np.log(data.close))
        volatility = np.std(returns) * np.sqrt(252)

        if abs(current_signal) < self.signal_threshold:
            direction = 0
            reason = (
                f"Signal {current_signal:.2f} below threshold {self.signal_threshold}"
            )
        elif current_signal > 0:
            direction = 1
            reason = f"Bullish signal: {current_signal:.2f}"
        else:
            direction = -1
            reason = f"Bearish signal: {current_signal:.2f}"

        if self.risk_manager.should_reduce_risk(portfolio):
            direction = 0
            reason = f"Risk reduction mode (drawdown: {portfolio.current_drawdown:.1%})"

        position = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=abs(current_signal),
            price=current_price,
            volatility=volatility,
            portfolio=portfolio,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
        )

        signal_values = {name: signal.get_current() for name, signal in signals.items()}

        return TradeDecision(
            symbol=symbol,
            direction=direction,
            position_size=position,
            signals=signal_values,
            combined_signal=current_signal,
            timestamp=datetime.now(),
            reason=reason,
        )
