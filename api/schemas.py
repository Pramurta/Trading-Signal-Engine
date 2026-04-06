"""Pydantic models for API request/response validation."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DataSourceEnum(StrEnum):
    synthetic = "synthetic"
    yfinance = "yfinance"


# --- Signal models ---


class SignalRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, examples=[["AAPL", "GOOGL"]])
    days: int = Field(default=252, ge=30, le=1000)
    source: DataSourceEnum = DataSourceEnum.synthetic


class CurrentSignals(BaseModel):
    zscore: float
    rsi: float
    macd: float
    bollinger: float
    combined: float
    direction: str


class SymbolSignals(BaseModel):
    current: CurrentSignals
    series: dict[str, list[float]]


class SignalResponse(BaseModel):
    symbols: dict[str, SymbolSignals]


# --- Backtest models ---


class BacktestRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, examples=[["AAPL", "GOOGL", "MSFT"]])
    days: int = Field(default=252, ge=30, le=1000)
    initial_capital: float = Field(default=100_000, ge=1000)
    source: DataSourceEnum = DataSourceEnum.synthetic


class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    volatility: float
    var_95: float


class BacktestResponse(BaseModel):
    backtest_id: str
    metrics: BacktestMetrics
    equity_curve: list[float]
