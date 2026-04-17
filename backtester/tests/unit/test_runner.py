"""Tests for backtester.runner - tick loop + timeout + traderData round-trip.

These tests build a small in-memory ``DayData`` (bypassing CSV) and feed a
hand-rolled trader class to the runner.
"""

from __future__ import annotations

import time

import pytest

from backtester.data_loader import DayData, PriceRow, TickSnapshot
from backtester.datamodel import Order, TradingState
from backtester.runner import BacktestConfig, RunResult, run_backtest

pytestmark = pytest.mark.unit


# ---------- helpers ----------

def _single_product_day(
    product: str = "K",
    ticks: list[int] | None = None,
    bid: int = 9998,
    ask: int = 10002,
    depth: int = 5,
) -> DayData:
    if ticks is None:
        ticks = [0, 100, 200]
    snapshots = {}
    for ts in ticks:
        row = PriceRow(
            day=0,
            timestamp=ts,
            product=product,
            bid_prices=(bid,),
            bid_volumes=(depth,),
            ask_prices=(ask,),
            ask_volumes=(depth,),
            mid_price=(bid + ask) / 2,
            profit_loss=0.0,
        )
        snapshots[ts] = TickSnapshot(
            timestamp=ts, prices={product: row}, market_trades={}
        )
    return DayData(products=[product], snapshots=snapshots)


class _NullTrader:
    """Trader that never trades. Used to smoke-test the tick loop."""

    def run(self, state: TradingState):
        return ({}, 0, "")


class _TakerTrader:
    """Trader that buys 1 lot at the ask on every tick."""

    def __init__(self, symbol: str = "K", price: int = 10010, qty: int = 1):
        self.symbol = symbol
        self.price = price
        self.qty = qty

    def run(self, state: TradingState):
        return ({self.symbol: [Order(self.symbol, self.price, self.qty)]}, 0, "")


class _SlowTrader:
    """Trader that sleeps over the timeout threshold."""

    def __init__(self, sleep_ms: int = 950):
        self.sleep_ms = sleep_ms

    def run(self, state: TradingState):
        time.sleep(self.sleep_ms / 1000)
        return ({"K": [Order("K", 10010, 1)]}, 0, "")


class _RaisingTrader:
    def run(self, state: TradingState):
        raise RuntimeError("boom")


class _StatefulTrader:
    """Increments a counter in traderData each tick. Tests round-trip."""

    def run(self, state: TradingState):
        import json
        payload = json.loads(state.traderData) if state.traderData else {"n": 0}
        payload["n"] += 1
        return ({}, 0, json.dumps(payload))


class _BadTraderDataTrader:
    """Returns non-JSON-serializable traderData -> should fail gracefully."""

    def run(self, state: TradingState):
        class NotSerializable:
            pass
        return ({}, 0, NotSerializable())  # type: ignore[return-value]


# ---------- tests ----------

class TestTickLoopBasics:
    def test_null_trader_completes(self) -> None:
        data = _single_product_day()
        result = run_backtest(
            _NullTrader(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        assert isinstance(result, RunResult)
        assert len(result.tick_logs) == 3
        assert result.final_positions == {"K": 0}
        for tick in result.tick_logs:
            assert tick.trades == []
            assert tick.position == 0

    def test_taker_trader_accumulates_long(self) -> None:
        data = _single_product_day()
        result = run_backtest(
            _TakerTrader(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        assert result.final_positions["K"] == 3
        # Each tick produced one trade at the ask (10002)
        prices = [t.price for log in result.tick_logs for t in log.trades]
        assert prices == [10002, 10002, 10002]


class TestTimeout:
    @pytest.mark.slow
    def test_slow_run_drops_orders(self) -> None:
        data = _single_product_day(ticks=[0])
        result = run_backtest(
            _SlowTrader(sleep_ms=950),
            data,
            BacktestConfig(position_limits={"K": 50}, timeout_ms=900),
        )
        log = result.tick_logs[0]
        assert log.trades == []
        assert log.position == 0
        assert any("timeout" in w.lower() for w in log.warnings)

    def test_yellow_warning_over_500ms(self) -> None:
        data = _single_product_day(ticks=[0])
        result = run_backtest(
            _SlowTrader(sleep_ms=550),
            data,
            BacktestConfig(position_limits={"K": 50}, timeout_ms=900, yellow_threshold_ms=500),
        )
        log = result.tick_logs[0]
        # Orders accepted (fill happened) but yellow warning flagged.
        assert any("slow" in w.lower() or "yellow" in w.lower() for w in log.warnings)


class TestException:
    def test_raising_trader_drops_orders(self) -> None:
        data = _single_product_day(ticks=[0])
        result = run_backtest(
            _RaisingTrader(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        log = result.tick_logs[0]
        assert log.trades == []
        assert log.position == 0
        assert any("exception" in w.lower() or "boom" in w.lower() for w in log.warnings)


class TestTraderDataRoundTrip:
    def test_state_persists_across_ticks(self) -> None:
        data = _single_product_day(ticks=[0, 100, 200, 300])
        result = run_backtest(
            _StatefulTrader(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        # Final traderData should show n=4.
        import json
        assert json.loads(result.final_trader_data)["n"] == 4

    def test_non_json_traderData_fails_gracefully(self) -> None:
        data = _single_product_day(ticks=[0])
        result = run_backtest(
            _BadTraderDataTrader(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        # Must not crash; must log a warning; traderData resets to empty string.
        assert result.final_trader_data == ""
        assert any(
            "trader" in w.lower() and "data" in w.lower()
            for w in result.tick_logs[0].warnings
        )

    def test_traderData_size_cap_enforced(self) -> None:
        """1MB cap per PRD. Over-size payload => reset to empty + warning."""

        class Chungus:
            def run(self, state: TradingState):
                # 2 MB string
                return ({}, 0, "x" * (2 * 1024 * 1024))

        data = _single_product_day(ticks=[0])
        result = run_backtest(
            Chungus(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        assert result.final_trader_data == ""
        assert any("size" in w.lower() for w in result.tick_logs[0].warnings)


class TestStateContent:
    def test_own_trades_populated_for_last_trade(self) -> None:
        """After a fill, state.own_trades for the next tick must include it."""

        class CaptureOwnTrades:
            seen: list[list] = []

            def run(self, state: TradingState):
                # Record what we see in own_trades at the start of this tick.
                CaptureOwnTrades.seen.append(list(state.own_trades.get("K", [])))
                if state.timestamp == 0:
                    return ({"K": [Order("K", 10010, 1)]}, 0, "")
                return ({}, 0, "")

        data = _single_product_day(ticks=[0, 100])
        run_backtest(
            CaptureOwnTrades(),
            data,
            BacktestConfig(position_limits={"K": 50}),
        )
        # Tick 0: no own trades yet (first tick)
        assert CaptureOwnTrades.seen[0] == []
        # Tick 100: own_trades should reflect the tick-0 fill.
        assert len(CaptureOwnTrades.seen[1]) == 1
        assert CaptureOwnTrades.seen[1][0].price == 10002
