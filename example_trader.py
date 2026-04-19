"""Deterministic sample market-maker used by smoke tests and fixtures.

Strategy (first principles):
- RESIN is pegged at 10000 (DGP assumption from Prosperity 3 rounds).
- Quote symmetric +/- 1 around 10000.
- Size: 5 per side.
- Inventory skew: if position > +HEAVY, quote only the sell side. If position
  < -HEAVY, quote only the buy side. This prevents runaway inventory.
- Every parameter has a first-principles justification and derives directly
  from the pegged-asset DGP. No moving averages, no z-scores, no indicators.

Zero randomness. Identical input => identical output.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from backtester.datamodel import Order, Symbol, TradingState


RESIN_SYMBOL = "RESIN"
PEG = 10000
QUOTE_SIZE = 5
HEAVY = 40  # inventory threshold beyond which we stop quoting the same side


class Trader:
    """Prosperity-compatible trader. Instantiated once per backtest.

    Round 2 support: exposes ``get_maf(state) -> float`` returning a
    constructor-configured bid. Default is ``0.0`` (do not bid). A trader
    that wants to participate in the MAF auction passes ``maf_bid=X`` at
    construction. The method returns a constant here for determinism; a
    production trader would compute the bid based on observed edge.
    """

    def __init__(
        self,
        peg: int = PEG,
        quote_size: int = QUOTE_SIZE,
        heavy: int = HEAVY,
        maf_bid: float = 0.0,
    ) -> None:
        self.peg = peg
        self.quote_size = quote_size
        self.heavy = heavy
        self.maf_bid = maf_bid

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        orders: Dict[Symbol, List[Order]] = {}
        symbol = RESIN_SYMBOL
        if symbol not in state.order_depths:
            return orders, 0, state.traderData

        position = state.position.get(symbol, 0)
        product_orders: List[Order] = []

        # Only quote the buy side if we're not overweight long.
        if position < self.heavy:
            product_orders.append(Order(symbol, self.peg - 1, self.quote_size))
        # Only quote the sell side if we're not overweight short.
        if position > -self.heavy:
            product_orders.append(Order(symbol, self.peg + 1, -self.quote_size))

        if product_orders:
            orders[symbol] = product_orders

        return orders, 0, state.traderData

    def get_maf(self, state: TradingState) -> float:
        """Declare the MAF for the Round 2 auction. Default method name.

        Returns the constructor-configured ``maf_bid`` as a SeaShells value.
        A trader could compute this dynamically from ``state`` (e.g. bid
        higher when edge is high) - the runner calls this once per tick.
        """
        return float(self.maf_bid)
