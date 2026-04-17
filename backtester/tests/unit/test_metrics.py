"""Tests for backtester.metrics - PnL, drawdown, fill quality."""

from __future__ import annotations

import pytest

from backtester.datamodel import Trade
from backtester.metrics import (
    PnLPoint,
    compute_drawdown,
    compute_metrics,
    compute_pnl_curve,
    compute_sharpe,
)
from backtester.runner import RunResult, TickLog

pytestmark = pytest.mark.unit


def _log(
    ts: int, trades: list[Trade], positions: dict[str, int], mids: dict[str, float]
) -> TickLog:
    pos = positions.get(next(iter(positions), ""), 0) if positions else 0
    return TickLog(
        timestamp=ts,
        duration_ms=0.0,
        trades=list(trades),
        position=pos,
        positions=dict(positions),
        warnings=[],
        rejections=[],
        mid_prices=dict(mids),
    )


class TestPnLCurve:
    def test_flat_with_no_trades(self) -> None:
        logs = [
            _log(0, [], {"K": 0}, {"K": 100.0}),
            _log(100, [], {"K": 0}, {"K": 101.0}),
        ]
        result = RunResult(
            tick_logs=logs, final_positions={"K": 0}, final_trader_data="", products=["K"]
        )
        curve = compute_pnl_curve(result)
        assert all(p.pnl == 0.0 for p in curve)

    def test_buy_then_price_rises(self) -> None:
        logs = [
            _log(
                0,
                [Trade("K", 100, 5, "SUBMISSION", "", 0)],
                {"K": 5},
                {"K": 100.0},
            ),
            _log(100, [], {"K": 5}, {"K": 110.0}),
        ]
        result = RunResult(
            tick_logs=logs, final_positions={"K": 5}, final_trader_data="", products=["K"]
        )
        curve = compute_pnl_curve(result)
        assert isinstance(curve[0], PnLPoint)
        # Tick 0: bought 5 @ 100, mark-to-mid @ 100 -> PnL = 0
        assert curve[0].pnl == pytest.approx(0.0)
        # Tick 1: hold 5, mid = 110 -> PnL = 5 * (110 - 100) = 50
        assert curve[1].pnl == pytest.approx(50.0)

    def test_sell_then_price_drops(self) -> None:
        logs = [
            _log(
                0,
                [Trade("K", 100, 5, "", "SUBMISSION", 0)],
                {"K": -5},
                {"K": 100.0},
            ),
            _log(100, [], {"K": -5}, {"K": 90.0}),
        ]
        result = RunResult(
            tick_logs=logs, final_positions={"K": -5}, final_trader_data="", products=["K"]
        )
        curve = compute_pnl_curve(result)
        assert curve[1].pnl == pytest.approx(50.0)

    def test_realized_and_unrealized_additive(self) -> None:
        """Buy 5 @ 100, sell 3 @ 110, hold 2. Mid at end = 108."""
        logs = [
            _log(
                0,
                [Trade("K", 100, 5, "SUBMISSION", "", 0)],
                {"K": 5},
                {"K": 100.0},
            ),
            _log(
                100,
                [Trade("K", 110, 3, "", "SUBMISSION", 100)],
                {"K": 2},
                {"K": 110.0},
            ),
            _log(200, [], {"K": 2}, {"K": 108.0}),
        ]
        result = RunResult(
            tick_logs=logs, final_positions={"K": 2}, final_trader_data="", products=["K"]
        )
        curve = compute_pnl_curve(result)
        # Tick 2 cash = -5*100 + 3*110 = -170. Inventory value = 2 * 108 = 216.
        # Total = 46.
        assert curve[-1].pnl == pytest.approx(46.0)


class TestDrawdown:
    def test_monotone_up_is_zero(self) -> None:
        curve = [PnLPoint(0, 0.0), PnLPoint(1, 10.0), PnLPoint(2, 20.0)]
        assert compute_drawdown(curve) == pytest.approx(0.0)

    def test_max_drawdown(self) -> None:
        curve = [
            PnLPoint(0, 0.0),
            PnLPoint(1, 50.0),
            PnLPoint(2, 30.0),
            PnLPoint(3, 80.0),
            PnLPoint(4, 10.0),
        ]
        # max drawdown = 80 - 10 = 70
        assert compute_drawdown(curve) == pytest.approx(70.0)


class TestSharpe:
    def test_zero_variance_zero_sharpe(self) -> None:
        curve = [PnLPoint(i, 0.0) for i in range(10)]
        assert compute_sharpe(curve) == 0.0

    def test_positive_drift_positive_sharpe(self) -> None:
        # Noisy upward drift: dpnl alternates {2, 0.5, 2, 0.5, ...} -> mean > 0, var > 0
        pnl = 0.0
        curve = [PnLPoint(0, 0.0)]
        for i in range(100):
            pnl += 2.0 if i % 2 == 0 else 0.5
            curve.append(PnLPoint(i + 1, pnl))
        assert compute_sharpe(curve) > 0.0


class TestComputeMetrics:
    def test_rollup(self) -> None:
        logs = [
            _log(0, [Trade("K", 100, 1, "SUBMISSION", "", 0)], {"K": 1}, {"K": 100.0}),
            _log(100, [], {"K": 1}, {"K": 110.0}),
        ]
        result = RunResult(
            tick_logs=logs, final_positions={"K": 1}, final_trader_data="", products=["K"]
        )
        m = compute_metrics(result)
        assert m.final_pnl == pytest.approx(10.0)
        assert m.max_drawdown == pytest.approx(0.0)
        assert m.num_trades == 1
        assert m.sharpe is not None
