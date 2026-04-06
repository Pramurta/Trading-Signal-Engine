"""Signal generation endpoints."""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    CurrentSignals,
    DataSourceEnum,
    SignalRequest,
    SignalResponse,
    SymbolSignals,
)
from src.data_handler import DataSource, MarketDataHandler
from src.signals import SignalEngine

router = APIRouter(prefix="/signals", tags=["signals"])


@router.post("/generate", response_model=SignalResponse)
async def generate_signals(request: SignalRequest):
    source = (
        DataSource.YFINANCE
        if request.source == DataSourceEnum.yfinance
        else DataSource.SYNTHETIC
    )

    try:
        handler = MarketDataHandler(source=source, random_seed=42)
        market_data = await handler.stream_market_data(
            request.symbols, days=request.days
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    engine = SignalEngine()
    result = {}

    for symbol, data in market_data.items():
        prices = data.close
        zscore = engine.calculate_zscore(prices, window=20)
        rsi = engine.calculate_rsi(prices, window=14)
        macd = engine.calculate_macd(prices)
        bollinger, _, _, _ = engine.calculate_bollinger_bands(prices)

        combined = engine.combine_signals(
            {"zscore": zscore, "rsi": rsi, "macd": macd, "bollinger": bollinger},
            weights={"zscore": 0.35, "rsi": 0.25, "macd": 0.25, "bollinger": 0.15},
        )

        direction_map = {1: "long", -1: "short", 0: "neutral"}

        result[symbol] = SymbolSignals(
            current=CurrentSignals(
                zscore=round(float(zscore.get_current()), 4),
                rsi=round(float(rsi.get_current()), 4),
                macd=round(float(macd.get_current()), 4),
                bollinger=round(float(bollinger.get_current()), 4),
                combined=round(float(combined.get_current()), 4),
                direction=direction_map[combined.get_direction()],
            ),
            series={
                "zscore": [round(float(v), 4) for v in zscore.values],
                "rsi": [round(float(v), 4) for v in rsi.values],
                "macd": [round(float(v), 4) for v in macd.values],
                "bollinger": [round(float(v), 4) for v in bollinger.values],
                "combined": [round(float(v), 4) for v in combined.values],
            },
        )

    return SignalResponse(symbols=result)
