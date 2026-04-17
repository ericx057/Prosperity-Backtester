"""Pin the datamodel API contract. A trader submission MUST work with this."""

from __future__ import annotations

import json

import pytest

from backtester.datamodel import (
    ConversionObservation,
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Trade,
    TradingState,
)

pytestmark = pytest.mark.unit


class TestOrder:
    def test_construction_positional(self) -> None:
        o = Order("RESIN", 10000, 5)
        assert o.symbol == "RESIN"
        assert o.price == 10000
        assert o.quantity == 5

    def test_repr_equals_str(self) -> None:
        o = Order("K", 12, -3)
        assert repr(o) == str(o)
        assert "K" in str(o)
        assert "12" in str(o)
        assert "-3" in str(o)


class TestOrderDepth:
    def test_empty_init(self) -> None:
        od = OrderDepth()
        assert od.buy_orders == {}
        assert od.sell_orders == {}

    def test_sell_orders_stored_negative(self) -> None:
        # Convention: sell_orders map price -> NEGATIVE quantity.
        od = OrderDepth()
        od.sell_orders[10004] = -5
        od.buy_orders[9998] = 3
        assert od.sell_orders[10004] < 0
        assert od.buy_orders[9998] > 0


class TestTrade:
    def test_minimum_construction(self) -> None:
        t = Trade("K", 100, 5)
        assert t.symbol == "K"
        assert t.price == 100
        assert t.quantity == 5
        assert t.buyer is None
        assert t.seller is None
        assert t.timestamp == 0

    def test_full_construction(self) -> None:
        t = Trade("K", 100, 5, "SUBMISSION", "NPC", 1500)
        assert t.buyer == "SUBMISSION"
        assert t.seller == "NPC"
        assert t.timestamp == 1500


class TestListing:
    def test_fields(self) -> None:
        listing = Listing("KELP", "KELP", 1)
        assert listing.symbol == "KELP"
        assert listing.product == "KELP"
        assert listing.denomination == 1


class TestConversionObservation:
    def test_mixedCase_fields_preserved(self) -> None:
        # Spec requires bidPrice/askPrice mixedCase; do NOT snake_case them.
        co = ConversionObservation(
            bidPrice=1.0,
            askPrice=2.0,
            transportFees=0.1,
            exportTariff=0.01,
            importTariff=0.02,
            sugarPrice=100.0,
            sunlightIndex=50.0,
        )
        assert co.bidPrice == 1.0
        assert co.askPrice == 2.0


class TestObservation:
    def test_empty(self) -> None:
        o = Observation({}, {})
        assert o.plainValueObservations == {}
        assert o.conversionObservations == {}

    def test_str_no_crash(self) -> None:
        o = Observation({"X": 1}, {})
        assert isinstance(str(o), str)


class TestTradingState:
    def test_construction_all_fields(self) -> None:
        ts = TradingState(
            traderData="",
            timestamp=0,
            listings={},
            order_depths={},
            own_trades={},
            market_trades={},
            position={},
            observations=Observation({}, {}),
        )
        assert ts.timestamp == 0
        assert ts.traderData == ""

    def test_toJSON_runs(self) -> None:
        ts = TradingState(
            traderData="",
            timestamp=100,
            listings={"K": Listing("K", "K", 1)},
            order_depths={"K": OrderDepth()},
            own_trades={},
            market_trades={},
            position={"K": 0},
            observations=Observation({}, {}),
        )
        payload = ts.toJSON()
        assert isinstance(payload, str)
        parsed = json.loads(payload)
        assert parsed["timestamp"] == 100


class TestProsperityEncoder:
    def test_encodes_order(self) -> None:
        o = Order("K", 10, 1)
        payload = json.dumps(o, cls=ProsperityEncoder)
        parsed = json.loads(payload)
        assert parsed == {"symbol": "K", "price": 10, "quantity": 1}
