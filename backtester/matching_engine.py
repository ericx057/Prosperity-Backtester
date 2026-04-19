"""Matching engine - pure, fidelity-critical.

Architectural invariant: this module has ZERO imports from runner, metrics,
reporter, sweeper, or data_loader. See tests/fidelity/test_module_boundary.py.

Public interface:

    engine = MatchingEngine(position_limits={"K": 50})
    result = engine.match(
        symbol, user_orders, book, position, market_trades, timestamp
    )
    # result is a MatchResult:
    #   .trades: list[Trade]
    #   .new_book: OrderDepth (fresh, never mutates the input)
    #   .new_position: int
    #   .rejections: list[str]

Fidelity rules (PRD section 4):

1. Match against the tick-start book. Callers pass the book as it exists AFTER
   ``prepare_state`` has populated it from historical data. That's the book
   user orders hit.
2. Crossing orders take (fill at book price). Non-crossing orders stay
   passive; they fill ONLY when a market-trade event for the tick passes
   through their price.
3. Aggregate position-limit check: sum(long qty) + position > limit OR
   position - sum(|short qty|) < -limit => reject the ENTIRE product batch.
4. Invalid orders (zero quantity, non-int fields) are rejected before any
   matching.
5. Self-cross: if the batch contains both a buy and a sell whose prices
   cross, reject the entire batch (PRD default; see tests).

Market-trade convention (follows jmerle reference):

- Crossing against the book => fill price = book (resting) price, buyer/seller
  tag: "SUBMISSION" on the user's side, "" on the opposite.
- Fill via market-trade pass-through => fill price = user's order price (they
  get the improvement vs. the NPC trade), buyer/seller tag: "SUBMISSION" on
  the user's side and the NPC's counterparty on the other side.

Volume-multiplier hook (Round 2 support):

``match`` accepts an optional ``volume_multiplier`` (default 1.0) which scales
the visible book depth. The engine itself is policy-free - the caller decides
when/why to boost (e.g., Round 2 auction win). The multiplier is applied to a
fresh book copy; the input book is never mutated. A multiplier of 1.0 is a
no-op and produces byte-identical output to pre-Round-2 behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from backtester.datamodel import Order, OrderDepth, Symbol, Trade


@dataclass
class MatchResult:
    trades: List[Trade]
    new_book: OrderDepth
    new_position: int
    rejections: List[str] = field(default_factory=list)


@dataclass
class _MarketTradeView:
    """Mutable wrapper tracking remaining buy/sell capacity of a market trade."""

    trade: Trade
    buy_capacity: int   # NPC side buying => user can sell into it
    sell_capacity: int  # NPC side selling => user can buy from it


def _copy_book(book: OrderDepth, volume_multiplier: float = 1.0) -> OrderDepth:
    """Return a fresh OrderDepth with the same levels. Input is never mutated.

    If ``volume_multiplier`` differs from 1.0, the copied book's per-level
    volumes are scaled by it and rounded toward zero (``int(v * m)``). Levels
    that round to zero volume are dropped. A multiplier of 1.0 is an exact
    passthrough.

    The multiplier is a pure data transformation; the engine has no opinion on
    why the caller supplied it.
    """
    out = OrderDepth()
    if volume_multiplier == 1.0:
        out.buy_orders = dict(book.buy_orders)
        out.sell_orders = dict(book.sell_orders)
        return out
    for price, qty in book.buy_orders.items():
        new_qty = int(qty * volume_multiplier)
        if new_qty > 0:
            out.buy_orders[price] = new_qty
    for price, qty in book.sell_orders.items():
        # sell_orders values are negative; scale and round toward zero.
        new_qty = int(qty * volume_multiplier)
        if new_qty < 0:
            out.sell_orders[price] = new_qty
    return out


def _validate_order(order: Order) -> None:
    """Raise ValueError for type-invalid orders (matches jmerle type_check_orders).

    Note: integer subclass ``bool`` is rejected for safety.
    """
    if not isinstance(order.symbol, str):
        raise ValueError(f"Order symbol must be str, got {type(order.symbol)!r}")
    if not isinstance(order.price, int) or isinstance(order.price, bool):
        raise ValueError(f"Order price must be int, got {type(order.price)!r} for {order}")
    if not isinstance(order.quantity, int) or isinstance(order.quantity, bool):
        raise ValueError(f"Order quantity must be int, got {type(order.quantity)!r} for {order}")


class MatchingEngine:
    """Stateless (per-call) matching engine.

    Holds only static configuration (position_limits). All mutable state lives
    in the caller. Safe to share across ticks and strategies.
    """

    def __init__(self, position_limits: Dict[Symbol, int]):
        # Take a defensive copy so callers can't mutate under us.
        self._limits: Dict[Symbol, int] = dict(position_limits)

    # ---------- public API ----------

    def limit_for(self, symbol: Symbol) -> int:
        return self._limits[symbol]

    def match(
        self,
        *,
        symbol: Symbol,
        user_orders: Sequence[Order],
        book: OrderDepth,
        position: int,
        market_trades: Sequence[Trade],
        timestamp: int,
        volume_multiplier: float = 1.0,
    ) -> MatchResult:
        """Match ``user_orders`` against ``book`` and ``market_trades``.

        Returns a fresh ``MatchResult`` and never mutates input arguments.

        ``volume_multiplier`` (default 1.0) scales the visible depth of the
        book copy used for matching. See ``_copy_book``. Must be non-negative.
        """
        if volume_multiplier < 0:
            raise ValueError(
                f"volume_multiplier must be >= 0; got {volume_multiplier}"
            )
        rejections: List[str] = []
        new_book = _copy_book(book, volume_multiplier=volume_multiplier)

        # ----- type validation (raises, same as live runner) -----
        for order in user_orders:
            _validate_order(order)

        # ----- filter: zero quantity -----
        filtered: List[Order] = []
        for order in user_orders:
            if order.quantity == 0:
                rejections.append(
                    f"Order for {symbol} rejected: zero quantity"
                )
                continue
            filtered.append(order)

        if not filtered:
            return MatchResult(
                trades=[], new_book=new_book, new_position=position, rejections=rejections
            )

        # ----- self-cross check -----
        buys = [o for o in filtered if o.quantity > 0]
        sells = [o for o in filtered if o.quantity < 0]
        if buys and sells:
            max_buy = max(o.price for o in buys)
            min_sell = min(o.price for o in sells)
            if max_buy >= min_sell:
                rejections.append(
                    f"Orders for {symbol} rejected: self-cross "
                    f"(buy {max_buy} >= sell {min_sell})"
                )
                return MatchResult(
                    trades=[], new_book=new_book, new_position=position, rejections=rejections
                )

        # ----- aggregate position-limit check -----
        limit = self._limits.get(symbol)
        if limit is None:
            rejections.append(
                f"Orders for {symbol} rejected: no position limit configured"
            )
            return MatchResult(
                trades=[], new_book=new_book, new_position=position, rejections=rejections
            )
        total_long = sum(o.quantity for o in filtered if o.quantity > 0)
        total_short = sum(abs(o.quantity) for o in filtered if o.quantity < 0)
        if position + total_long > limit or position - total_short < -limit:
            rejections.append(
                f"Orders for {symbol} exceeded position limit of {limit}"
            )
            return MatchResult(
                trades=[], new_book=new_book, new_position=position, rejections=rejections
            )

        # ----- match in submission order (no internal sort) -----
        market_views = [
            _MarketTradeView(trade=t, buy_capacity=t.quantity, sell_capacity=t.quantity)
            for t in market_trades
        ]
        trades: List[Trade] = []
        cur_position = position
        for order in filtered:
            if order.quantity > 0:
                new_trades, cur_position = self._match_buy(
                    order, new_book, cur_position, market_views, timestamp
                )
            else:
                new_trades, cur_position = self._match_sell(
                    order, new_book, cur_position, market_views, timestamp
                )
            trades.extend(new_trades)

        return MatchResult(
            trades=trades,
            new_book=new_book,
            new_position=cur_position,
            rejections=rejections,
        )

    # ---------- private matching ----------

    def _match_buy(
        self,
        order: Order,
        book: OrderDepth,
        position: int,
        market_views: List[_MarketTradeView],
        timestamp: int,
    ) -> tuple[List[Trade], int]:
        trades: List[Trade] = []
        remaining = order.quantity

        # Walk asks in ASCENDING price order. Fill price = ask (resting) price.
        candidate_prices = sorted(p for p in book.sell_orders if p <= order.price)
        for ask_price in candidate_prices:
            if remaining <= 0:
                break
            available = abs(book.sell_orders[ask_price])
            volume = min(remaining, available)
            trades.append(
                Trade(
                    symbol=order.symbol,
                    price=ask_price,
                    quantity=volume,
                    buyer="SUBMISSION",
                    seller="",
                    timestamp=timestamp,
                )
            )
            remaining -= volume
            position += volume
            new_level = book.sell_orders[ask_price] + volume  # sell_orders are negative
            if new_level == 0:
                book.sell_orders.pop(ask_price)
            else:
                book.sell_orders[ask_price] = new_level

        # Market-trade pass-through: NPC sells at/below user's buy price.
        for mv in market_views:
            if remaining <= 0:
                break
            if mv.sell_capacity <= 0:
                continue
            if mv.trade.price > order.price:
                continue
            volume = min(remaining, mv.sell_capacity)
            trades.append(
                Trade(
                    symbol=order.symbol,
                    price=order.price,
                    quantity=volume,
                    buyer="SUBMISSION",
                    seller=mv.trade.seller or "",
                    timestamp=timestamp,
                )
            )
            mv.sell_capacity -= volume
            remaining -= volume
            position += volume

        return trades, position

    def _match_sell(
        self,
        order: Order,
        book: OrderDepth,
        position: int,
        market_views: List[_MarketTradeView],
        timestamp: int,
    ) -> tuple[List[Trade], int]:
        trades: List[Trade] = []
        remaining = abs(order.quantity)

        # Walk bids in DESCENDING price order. Fill price = bid (resting) price.
        candidate_prices = sorted(
            (p for p in book.buy_orders if p >= order.price), reverse=True
        )
        for bid_price in candidate_prices:
            if remaining <= 0:
                break
            available = book.buy_orders[bid_price]
            volume = min(remaining, available)
            trades.append(
                Trade(
                    symbol=order.symbol,
                    price=bid_price,
                    quantity=volume,
                    buyer="",
                    seller="SUBMISSION",
                    timestamp=timestamp,
                )
            )
            remaining -= volume
            position -= volume
            new_level = book.buy_orders[bid_price] - volume
            if new_level == 0:
                book.buy_orders.pop(bid_price)
            else:
                book.buy_orders[bid_price] = new_level

        # Market-trade pass-through: NPC buys at/above user's sell price.
        for mv in market_views:
            if remaining <= 0:
                break
            if mv.buy_capacity <= 0:
                continue
            if mv.trade.price < order.price:
                continue
            volume = min(remaining, mv.buy_capacity)
            trades.append(
                Trade(
                    symbol=order.symbol,
                    price=order.price,
                    quantity=volume,
                    buyer=mv.trade.buyer or "",
                    seller="SUBMISSION",
                    timestamp=timestamp,
                )
            )
            mv.buy_capacity -= volume
            remaining -= volume
            position -= volume

        return trades, position
