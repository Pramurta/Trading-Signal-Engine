"""Tests for backtest endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_run_backtest_synthetic():
    response = client.post(
        "/backtest/run",
        json={
            "symbols": ["AAPL", "MSFT"],
            "days": 100,
            "initial_capital": 100000,
            "source": "synthetic",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert "backtest_id" in data
    assert "metrics" in data
    assert "equity_curve" in data
    assert data["metrics"]["num_trades"] >= 0
    assert len(data["equity_curve"]) > 0


def test_get_backtest_report():
    # First run a backtest
    run_response = client.post(
        "/backtest/run",
        json={"symbols": ["AAPL"], "days": 100, "source": "synthetic"},
    )
    backtest_id = run_response.json()["backtest_id"]

    # Then retrieve it
    report_response = client.get(f"/backtest/report/{backtest_id}")
    assert report_response.status_code == 200
    assert report_response.json()["backtest_id"] == backtest_id


def test_get_backtest_report_not_found():
    response = client.get("/backtest/report/nonexistent-id")
    assert response.status_code == 404
