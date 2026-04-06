"""Backtest endpoints."""

import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BacktestMetrics,
    BacktestRequest,
    BacktestResponse,
    DataSourceEnum,
)
from src.data_handler import DataSource, MarketDataHandler
from src.risk_manager import RiskManager
from src.signals import SignalEngine
from src.strategy import TradingStrategy

router = APIRouter(prefix="/backtest", tags=["backtest"])

# In-memory cache for backtest results
_results_cache: dict[str, BacktestResponse] = {}


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    source = (
        DataSource.YFINANCE
        if request.source == DataSourceEnum.yfinance
        else DataSource.SYNTHETIC
    )

    handler = MarketDataHandler(source=source, random_seed=42)
    strategy = TradingStrategy(
        data_handler=handler,
        signal_engine=SignalEngine(),
        risk_manager=RiskManager(max_position_pct=0.15, max_drawdown_pct=0.20),
    )

    try:
        results = await strategy.run_backtest(
            symbols=request.symbols,
            days=request.days,
            initial_capital=request.initial_capital,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    backtest_id = str(uuid.uuid4())

    response = BacktestResponse(
        backtest_id=backtest_id,
        metrics=BacktestMetrics(
            total_return=round(float(results.total_return), 4),
            sharpe_ratio=round(float(results.sharpe_ratio), 4),
            max_drawdown=round(float(results.max_drawdown), 4),
            win_rate=round(float(results.win_rate), 4),
            profit_factor=round(float(results.profit_factor), 4),
            num_trades=int(results.num_trades),
            volatility=round(float(results.risk_metrics.volatility), 4),
            var_95=round(float(results.risk_metrics.var_95), 4),
        ),
        equity_curve=[round(float(v), 2) for v in results.equity_curve],
    )

    _results_cache[backtest_id] = response
    return response


@router.get("/report/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_report(backtest_id: str):
    if backtest_id not in _results_cache:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return _results_cache[backtest_id]
