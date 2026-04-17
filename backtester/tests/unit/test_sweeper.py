"""Tests for backtester.sweeper - grid search + heatmap."""

from __future__ import annotations

from pathlib import Path

import pytest

from backtester.data_loader import DayData, PriceRow, TickSnapshot
from backtester.sweeper import (
    SweepConfig,
    SweepParam,
    cartesian_combos,
    run_sweep,
    write_heatmap,
    write_sweep_csv,
)

pytestmark = pytest.mark.unit


def _synth_day() -> DayData:
    ticks = [0, 100, 200, 300, 400]
    snapshots = {}
    for ts in ticks:
        snapshots[ts] = TickSnapshot(
            timestamp=ts,
            prices={
                "K": PriceRow(
                    day=0,
                    timestamp=ts,
                    product="K",
                    bid_prices=(9998,),
                    bid_volumes=(5,),
                    ask_prices=(10002,),
                    ask_volumes=(5,),
                    mid_price=10000.0,
                    profit_loss=0.0,
                )
            },
            market_trades={},
        )
    return DayData(products=["K"], snapshots=snapshots)


class ParamTrader:
    """Trader whose behavior depends on instance-level params."""

    def __init__(self, threshold: int = 0, size: int = 1) -> None:
        self.threshold = threshold
        self.size = size

    def run(self, state):  # type: ignore[no-untyped-def]
        from backtester.datamodel import Order
        orders = {}
        if self.threshold >= 0:
            orders["K"] = [Order("K", 10010, self.size)]
        return orders, 0, ""


class TestCartesianCombos:
    def test_two_params(self) -> None:
        params = [
            SweepParam("threshold", [0, 1, 2]),
            SweepParam("size", [1, 3]),
        ]
        combos = cartesian_combos(params)
        assert len(combos) == 6
        assert {"threshold": 0, "size": 1} in combos
        assert {"threshold": 2, "size": 3} in combos

    def test_single_param(self) -> None:
        combos = cartesian_combos([SweepParam("a", [1, 2, 3])])
        assert combos == [{"a": 1}, {"a": 2}, {"a": 3}]


class TestSweep:
    def test_produces_row_per_combo(self, tmp_path: Path) -> None:
        data = _synth_day()
        cfg = SweepConfig(
            trader_factory=lambda **kw: ParamTrader(**kw),
            params=[
                SweepParam("threshold", [0, 1]),
                SweepParam("size", [1, 2]),
            ],
            position_limits={"K": 50},
            workers=1,
        )
        rows = run_sweep(cfg, data)
        assert len(rows) == 4
        for row in rows:
            assert "final_pnl" in row
            assert "max_drawdown" in row
            assert "sharpe" in row
            assert "threshold" in row
            assert "size" in row

    def test_deterministic(self, tmp_path: Path) -> None:
        data = _synth_day()
        cfg = SweepConfig(
            trader_factory=lambda **kw: ParamTrader(**kw),
            params=[SweepParam("size", [1, 2, 3])],
            position_limits={"K": 50},
            workers=1,
        )
        rows1 = run_sweep(cfg, data)
        rows2 = run_sweep(cfg, data)
        # Sort to avoid worker-ordering issues even though workers=1.
        rows1s = sorted(rows1, key=lambda r: r["size"])
        rows2s = sorted(rows2, key=lambda r: r["size"])
        assert [r["final_pnl"] for r in rows1s] == [r["final_pnl"] for r in rows2s]

    def test_parallel_matches_serial(self) -> None:
        data = _synth_day()
        params = [SweepParam("size", [1, 2, 3])]
        cfg_serial = SweepConfig(
            trader_factory=ParamTrader,
            params=params,
            position_limits={"K": 50},
            workers=1,
        )
        cfg_parallel = SweepConfig(
            trader_factory=ParamTrader,
            params=params,
            position_limits={"K": 50},
            workers=2,
        )
        rows_serial = sorted(run_sweep(cfg_serial, data), key=lambda r: r["size"])
        rows_parallel = sorted(run_sweep(cfg_parallel, data), key=lambda r: r["size"])
        assert [r["final_pnl"] for r in rows_serial] == [
            r["final_pnl"] for r in rows_parallel
        ]


class TestSweepOutputs:
    def test_csv_written(self, tmp_path: Path) -> None:
        rows = [
            {"a": 0, "b": 0, "final_pnl": 1.0, "max_drawdown": 0.0, "sharpe": 0.0, "num_trades": 1},
            {"a": 1, "b": 0, "final_pnl": 2.0, "max_drawdown": 0.5, "sharpe": 0.1, "num_trades": 2},
        ]
        out = tmp_path / "sweep.csv"
        write_sweep_csv(rows, out)
        assert out.exists()
        content = out.read_text()
        assert "a,b" in content or "final_pnl" in content
        assert "2.0" in content

    def test_heatmap_written(self, tmp_path: Path) -> None:
        rows = []
        for a in [0, 1, 2]:
            for b in [0, 1]:
                rows.append({"a": a, "b": b, "final_pnl": float(a * 10 + b)})
        out = tmp_path / "heatmap.png"
        write_heatmap(rows, "a", "b", "final_pnl", out)
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
