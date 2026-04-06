"""Tests for signals endpoint."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_generate_signals_synthetic():
    response = client.post(
        "/signals/generate",
        json={"symbols": ["AAPL"], "days": 100, "source": "synthetic"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data["symbols"]

    aapl = data["symbols"]["AAPL"]
    assert "current" in aapl
    assert "series" in aapl
    assert aapl["current"]["direction"] in ("long", "short", "neutral")


def test_generate_signals_multiple_symbols():
    response = client.post(
        "/signals/generate",
        json={"symbols": ["AAPL", "GOOGL"], "days": 100, "source": "synthetic"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data["symbols"]
    assert "GOOGL" in data["symbols"]


def test_generate_signals_series_length():
    days = 100
    response = client.post(
        "/signals/generate",
        json={"symbols": ["AAPL"], "days": days, "source": "synthetic"},
    )
    data = response.json()
    series = data["symbols"]["AAPL"]["series"]
    assert len(series["zscore"]) == days
    assert len(series["rsi"]) == days
