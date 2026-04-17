"""Tests for backtester.reporter: JSON + matplotlib 4-panel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtester.datamodel import Trade
from backtester.reporter import write_json, write_summary_plot
from backtester.runner import RunResult, TickLog

pytestmark = pytest.mark.unit


def _make_result() -> RunResult:
    logs = [
        TickLog(
            timestamp=0,
            duration_ms=1.2,
            trades=[Trade("K", 100, 5, "SUBMISSION", "", 0)],
            position=5,
            positions={"K": 5},
            warnings=[],
            rejections=[],
            mid_prices={"K": 100.0},
        ),
        TickLog(
            timestamp=100,
            duration_ms=0.8,
            trades=[Trade("K", 110, 3, "", "SUBMISSION", 100)],
            position=2,
            positions={"K": 2},
            warnings=["yellow: slow"],
            rejections=["K: limit"],
            mid_prices={"K": 108.0},
        ),
    ]
    return RunResult(
        tick_logs=logs, final_positions={"K": 2}, final_trader_data="", products=["K"]
    )


class TestJsonReport:
    def test_produces_valid_json(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "results.json"
        write_json(result, out)
        assert out.exists()
        doc = json.loads(out.read_text())
        assert "tick_logs" in doc
        assert "metrics" in doc
        assert "products" in doc

    def test_json_serializes_trades(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "results.json"
        write_json(result, out)
        doc = json.loads(out.read_text())
        tick0 = doc["tick_logs"][0]
        assert tick0["trades"][0]["price"] == 100
        assert tick0["trades"][0]["quantity"] == 5
        assert tick0["trades"][0]["buyer"] == "SUBMISSION"

    def test_metrics_block_present(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "results.json"
        write_json(result, out)
        doc = json.loads(out.read_text())
        m = doc["metrics"]
        assert "final_pnl" in m
        assert "max_drawdown" in m
        assert "num_trades" in m


class TestSummaryPlot:
    def test_writes_png(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "summary.png"
        write_summary_plot(result, out)
        assert out.exists()
        # Sanity: PNG magic bytes.
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
