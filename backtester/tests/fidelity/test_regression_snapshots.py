"""Regression-lock tests: snapshot matching engine outputs for fixed inputs.

These tests read expected output JSONs from ``tests/fidelity/snapshots/``.
If the matching engine changes behavior on a fixed fixture, these tests fail.
To intentionally update the snapshots, re-run with:

    pytest backtester/tests/fidelity/test_regression_snapshots.py --update-snapshots

CAUTION: updating snapshots is a deliberate act. It means you accept the
new matching behavior as the new baseline. Review the diff before committing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pytest

from backtester.datamodel import Order, OrderDepth, Trade
from backtester.matching_engine import MatchingEngine

pytestmark = pytest.mark.fidelity

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _result_to_dict(result: Any) -> Dict[str, Any]:
    return {
        "trades": [
            {
                "symbol": t.symbol,
                "price": t.price,
                "quantity": t.quantity,
                "buyer": t.buyer or "",
                "seller": t.seller or "",
                "timestamp": t.timestamp,
            }
            for t in result.trades
        ],
        "new_book": {
            "buy_orders": dict(result.new_book.buy_orders),
            "sell_orders": dict(result.new_book.sell_orders),
        },
        "new_position": result.new_position,
        "rejections": list(result.rejections),
    }


# Each scenario is a (name, inputs) tuple. Inputs reproduce a realistic
# matching scenario end-to-end.

SCENARIOS = [
    (
        "simple_buy_crosses_single_ask",
        {
            "symbol": "K",
            "user_orders": [Order("K", 10010, 3)],
            "book_buys": {9990: 5},
            "book_sells": {10004: -5},
            "position": 0,
            "market_trades": [],
            "timestamp": 100,
            "limit": 50,
        },
    ),
    (
        "buy_sweeps_three_ask_levels",
        {
            "symbol": "K",
            "user_orders": [Order("K", 10010, 9)],
            "book_buys": {9990: 5},
            "book_sells": {10004: -2, 10005: -3, 10007: -4},
            "position": 0,
            "market_trades": [],
            "timestamp": 100,
            "limit": 50,
        },
    ),
    (
        "aggregate_limit_rejected",
        {
            "symbol": "K",
            "user_orders": [Order("K", 10000, 30), Order("K", 10000, 30)],
            "book_buys": {},
            "book_sells": {10000: -100},
            "position": 0,
            "market_trades": [],
            "timestamp": 100,
            "limit": 50,
        },
    ),
    (
        "maker_buy_fills_via_market_trade",
        {
            "symbol": "K",
            "user_orders": [Order("K", 9998, 2)],
            "book_buys": {9990: 5},
            "book_sells": {10004: -5},
            "position": 0,
            "market_trades": [
                Trade("K", 9997, 2, buyer="NPC_A", seller="NPC_B", timestamp=100),
            ],
            "timestamp": 100,
            "limit": 50,
        },
    ),
    (
        "self_cross_rejected",
        {
            "symbol": "K",
            "user_orders": [Order("K", 10002, 3), Order("K", 10001, -3)],
            "book_buys": {9990: 5},
            "book_sells": {10004: -5},
            "position": 0,
            "market_trades": [],
            "timestamp": 100,
            "limit": 50,
        },
    ),
    (
        "short_walks_bid_ladder_down",
        {
            "symbol": "K",
            "user_orders": [Order("K", 9990, -6)],
            "book_buys": {9995: 3, 9998: 2, 10000: 1},
            "book_sells": {10010: -5},
            "position": 0,
            "market_trades": [],
            "timestamp": 100,
            "limit": 50,
        },
    ),
]


def _run_scenario(inputs: Dict[str, Any]) -> Dict[str, Any]:
    engine = MatchingEngine(position_limits={inputs["symbol"]: inputs["limit"]})
    book = OrderDepth()
    book.buy_orders = dict(inputs["book_buys"])
    book.sell_orders = dict(inputs["book_sells"])
    result = engine.match(
        symbol=inputs["symbol"],
        user_orders=inputs["user_orders"],
        book=book,
        position=inputs["position"],
        market_trades=inputs["market_trades"],
        timestamp=inputs["timestamp"],
    )
    return _result_to_dict(result)


@pytest.fixture
def update_snapshots(request: Any) -> bool:
    return bool(request.config.getoption("--update-snapshots", default=False))


@pytest.mark.parametrize("name,inputs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_snapshot(name: str, inputs: Dict[str, Any], update_snapshots: bool) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    current = _run_scenario(inputs)
    current_text = json.dumps(current, indent=2, sort_keys=True)
    if update_snapshots or not path.exists():
        path.write_text(current_text)
        assert current_text == current_text
        return
    # Compare the canonicalized JSON texts so int->string key coercion
    # is normalized on both sides.
    expected_text = json.dumps(
        json.loads(path.read_text()), indent=2, sort_keys=True
    )
    # Round-trip ``current`` through JSON too to coerce int keys to strings.
    current_canonical = json.dumps(
        json.loads(current_text), indent=2, sort_keys=True
    )
    assert current_canonical == expected_text, (
        f"Snapshot mismatch for {name}. Re-run with --update-snapshots "
        "to accept (and commit the diff)."
    )
