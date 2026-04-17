"""Fidelity fixture tests for the matching engine.

These pin the matching engine against the live Prosperity simulator. They are
the load-bearing artifact of this project. Any change to the matching engine
that breaks them MUST be reviewed — a break indicates the offline engine has
drifted from live mechanics.

Invariants (from PRD section 4):
1. Within-tick sequence: prev user orders cancelled -> NPC book populates ->
   user orders match against resulting book.
2. Maker vs taker: crossing => taker, non-crossing => maker (no mid-spread
   phantom fills).
3. Fill price = resting order's price, not user's.
4. Aggregate limit check: partial-fill scenarios that could breach => entire
   batch for THAT PRODUCT rejected.
5. traderData forced through json.dumps/loads every tick (tested via runner).
6. Timeout / exception in run() => drop orders, position unchanged.
"""

from __future__ import annotations

import pytest

from backtester.datamodel import Order, OrderDepth, Trade
from backtester.matching_engine import MatchingEngine, MatchResult

pytestmark = pytest.mark.fidelity


# ---------- helpers ----------

def make_depth(bids: dict[int, int] | None = None, asks: dict[int, int] | None = None) -> OrderDepth:
    """Build an OrderDepth. ``asks`` values should be NEGATIVE per spec."""
    od = OrderDepth()
    if bids:
        od.buy_orders = dict(bids)
    if asks:
        od.sell_orders = dict(asks)
    return od


# ---------- taker: fill price = book price, walk levels ----------

class TestTakerFillPrice:
    def test_buy_crossing_single_ask_level_fills_at_ask(self) -> None:
        """User buy @ 10010 against ask at 10004 -> fill @ 10004, not 10010."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={9990: 5}, asks={10004: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10010, 3)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.price == 10004, "fill price must be book price, not user's"
        assert trade.quantity == 3
        assert trade.buyer == "SUBMISSION"
        assert trade.seller == ""
        assert result.new_position == 3
        assert result.rejections == []

    def test_buy_crosses_3_ask_levels_fills_at_each_ask_price(self) -> None:
        """User sweeps 3 levels -> 3 trades, each at its own ask price."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(
            bids={9990: 5},
            asks={10004: -2, 10005: -3, 10007: -4},
        )
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10010, 9)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        # walk in ascending ask price order
        prices_filled = [t.price for t in result.trades]
        qtys = [t.quantity for t in result.trades]
        assert prices_filled == [10004, 10005, 10007]
        assert qtys == [2, 3, 4]
        assert result.new_position == 9

    def test_sell_crossing_walks_bids_descending(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(
            bids={9995: 3, 9998: 2, 10000: 1},
            asks={10010: -5},
        )
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 9990, -6)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        # sells walk bids in DESCENDING price order
        prices_filled = [t.price for t in result.trades]
        qtys = [t.quantity for t in result.trades]
        assert prices_filled == [10000, 9998, 9995]
        assert qtys == [1, 2, 3]
        assert result.new_position == -6
        for t in result.trades:
            assert t.buyer == ""
            assert t.seller == "SUBMISSION"

    def test_partial_fill_when_depth_exhausted(self) -> None:
        """If user quantity > total book depth, fill what we can; no phantom."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10004: -3})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10010, 10)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert len(result.trades) == 1
        assert result.trades[0].quantity == 3
        assert result.new_position == 3


# ---------- maker: no phantom fills in spread ----------

class TestMakerNoPhantomFills:
    def test_buy_below_best_ask_does_not_fill_from_book(self) -> None:
        """Quoting inside spread without market trades => NO fill."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={9990: 5}, asks={10004: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 9995, 3)],  # inside spread, below ask
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.trades == []
        assert result.new_position == 0

    def test_sell_above_best_bid_does_not_fill_from_book(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={9990: 5}, asks={10004: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, -3)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.trades == []
        assert result.new_position == 0


# ---------- maker fills via market-trade pass-through ----------

class TestMakerFillsViaMarketTrades:
    def test_maker_buy_fills_when_npc_sells_through(self) -> None:
        """User maker buy at 9998. NPC sells at 9997 -> user eats that flow."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={9990: 5}, asks={10004: -5})
        # A market trade priced 9997 with a seller implies NPC hit bid at 9997.
        # Since user is bidding 9998 (better), they should front-run NPC buyer.
        npc_trade = Trade("K", 9997, 2, buyer="NPC_A", seller="NPC_B", timestamp=100)
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 9998, 2)],
            book=od,
            position=0,
            market_trades=[npc_trade],
            timestamp=100,
        )
        assert len(result.trades) == 1
        assert result.trades[0].quantity == 2
        assert result.trades[0].buyer == "SUBMISSION"
        # User's price is what they post; the market trade at 9997 would have
        # happened at some price anyway — fill price follows the jmerle convention:
        # user's order price (they got the improvement by posting at 9998).
        assert result.trades[0].price == 9998
        assert result.new_position == 2

    def test_maker_sell_fills_when_npc_buys_through(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={9990: 5}, asks={10004: -5})
        npc_trade = Trade("K", 10002, 3, buyer="NPC_A", seller="NPC_B", timestamp=100)
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, -3)],
            book=od,
            position=0,
            market_trades=[npc_trade],
            timestamp=100,
        )
        assert len(result.trades) == 1
        assert result.trades[0].quantity == 3
        assert result.trades[0].seller == "SUBMISSION"
        assert result.new_position == -3


# ---------- aggregate position-limit batch rejection ----------

class TestAggregateLimitBatchRejection:
    def test_two_buys_totaling_limit_breach_reject_whole_product(self) -> None:
        """Two orders each <= limit but together > limit -> both rejected."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10000: -100})
        # Position 0, sum of longs = 60 > limit 50 => reject all K orders.
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, 30), Order("K", 10000, 30)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.trades == []
        assert result.new_position == 0
        assert len(result.rejections) >= 1
        assert "K" in result.rejections[0]
        assert "limit" in result.rejections[0].lower()

    def test_one_buy_at_exact_limit_accepted(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10000: -100})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, 50)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.new_position == 50
        assert result.rejections == []

    def test_sell_side_aggregate_breach_rejected(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={10000: 100})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, -30), Order("K", 10000, -30)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.trades == []
        assert result.new_position == 0
        assert len(result.rejections) >= 1

    def test_existing_position_tight_limit(self) -> None:
        """Already long 40 on limit 50 -> can only buy 10 more."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10000: -100})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, 20)],
            book=od,
            position=40,
            market_trades=[],
            timestamp=100,
        )
        # 40 + 20 = 60 > 50 -> whole batch rejected.
        assert result.trades == []
        assert result.new_position == 40


# ---------- zero or negative quantity rejection ----------

class TestInvalidOrders:
    def test_zero_quantity_rejected(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10000: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10000, 0)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.trades == []
        assert any("quantity" in r.lower() for r in result.rejections)

    def test_non_int_price_raises(self) -> None:
        """Matches jmerle runner's ValueError on float price."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10000: -5})
        with pytest.raises(ValueError):
            engine.match(
                symbol="K",
                user_orders=[Order("K", 10000.5, 1)],  # type: ignore[arg-type]
                book=od,
                position=0,
                market_trades=[],
                timestamp=100,
            )

    def test_non_int_quantity_raises(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10000: -5})
        with pytest.raises(ValueError):
            engine.match(
                symbol="K",
                user_orders=[Order("K", 10000, 1.5)],  # type: ignore[arg-type]
                book=od,
                position=0,
                market_trades=[],
                timestamp=100,
            )


# ---------- self-cross handling ----------

class TestSelfCross:
    def test_user_buy_and_sell_same_tick_both_rejected(self) -> None:
        """PRD section 11 default: reject both. Document and pin.

        Rationale: the live exchange does NOT internally cross two user orders.
        Live behavior cannot be reliably reproduced if we self-trade, so we
        reject. Both legs remain unfilled; position unchanged.
        """
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(bids={9990: 5}, asks={10004: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10002, 3), Order("K", 10001, -3)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.trades == []
        assert result.new_position == 0
        assert any("self" in r.lower() or "cross" in r.lower() for r in result.rejections)


# ---------- book state after match ----------

class TestBookStateAfterMatch:
    def test_consumed_level_removed_from_new_book(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10004: -3, 10005: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10010, 3)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert 10004 not in result.new_book.sell_orders
        assert result.new_book.sell_orders[10005] == -5

    def test_partial_level_reduced(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10004: -5})
        result = engine.match(
            symbol="K",
            user_orders=[Order("K", 10010, 3)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert result.new_book.sell_orders[10004] == -2

    def test_original_book_not_mutated(self) -> None:
        """The passed-in book must remain intact — pure function invariant."""
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth(asks={10004: -5})
        original = dict(od.sell_orders)
        engine.match(
            symbol="K",
            user_orders=[Order("K", 10010, 3)],
            book=od,
            position=0,
            market_trades=[],
            timestamp=100,
        )
        assert od.sell_orders == original


# ---------- MatchResult shape ----------

class TestMatchResultShape:
    def test_empty_input_produces_empty_result(self) -> None:
        engine = MatchingEngine(position_limits={"K": 50})
        od = make_depth()
        result = engine.match(
            symbol="K",
            user_orders=[],
            book=od,
            position=5,
            market_trades=[],
            timestamp=100,
        )
        assert isinstance(result, MatchResult)
        assert result.trades == []
        assert result.new_position == 5
        assert result.rejections == []
