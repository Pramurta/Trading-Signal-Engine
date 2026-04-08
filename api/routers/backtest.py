"""Backtest endpoints."""

import uuid

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    BacktestMetrics,
    BacktestRequest,
    BacktestResponse,
    DataSourceEnum,
)
from src.backtest import BacktestConfig, Backtester
from src.data_handler import DataSource, MarketDataHandler
from src.signals import SignalEngine

router = APIRouter(prefix="/backtest", tags=["backtest"])

_results_cache: dict[str, BacktestResponse] = {}


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    source = (
        DataSource.YFINANCE
        if request.source == DataSourceEnum.yfinance
        else DataSource.SYNTHETIC
    )

    handler = MarketDataHandler(source=source, random_seed=42)

    try:
        market_data = await handler.stream_market_data(
            request.symbols, days=request.days
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not market_data:
        raise HTTPException(status_code=400, detail="No data returned for symbols")

    # Build price and signal matrices (T x N)
    engine = SignalEngine()
    min_length = min(len(d.close) for d in market_data.values())

    prices = np.column_stack([d.close[:min_length] for d in market_data.values()])
    signals = np.column_stack(
        [
            engine.combine_signals(
                {
                    "zscore": engine.calculate_zscore(d.close[:min_length]),
                    "rsi": engine.calculate_rsi(d.close[:min_length]),
                    "macd": engine.calculate_macd(d.close[:min_length]),
                    "bollinger": engine.calculate_bollinger_bands(d.close[:min_length])[
                        0
                    ],
                },
                weights={"zscore": 0.35, "rsi": 0.25, "macd": 0.25, "bollinger": 0.15},
            ).values
            for d in market_data.values()
        ]
    )

    config = BacktestConfig(initial_capital=request.initial_capital)
    backtester = Backtester(config)

    try:
        metrics, equity_curve, _ = backtester.run(prices, signals)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    backtest_id = str(uuid.uuid4())

    response = BacktestResponse(
        backtest_id=backtest_id,
        metrics=BacktestMetrics(
            total_return=round(float(metrics.total_return), 4),
            sharpe_ratio=round(float(metrics.sharpe_ratio), 4),
            max_drawdown=round(float(metrics.max_drawdown), 4),
            win_rate=round(float(metrics.win_rate), 4),
            profit_factor=round(float(metrics.profit_factor), 4),
            num_trades=int(metrics.num_trades),
            volatility=round(float(metrics.volatility), 4),
            var_95=round(
                float(np.percentile(np.diff(equity_curve) / equity_curve[:-1], 5))
                if len(equity_curve) > 1
                else 0.0,
                4,
            ),
        ),
        equity_curve=[round(float(v), 2) for v in equity_curve],
    )

    _results_cache[backtest_id] = response
    return response


@router.get("/report/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_report(backtest_id: str):
    if backtest_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return _results_cache[backtest_id]
