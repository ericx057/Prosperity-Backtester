"""CSV -> per-tick snapshots.

Prosperity historical data:
- prices_round_{N}_day_{M}.csv -- semicolon-separated, one row per (timestamp, product).
  Columns: day; timestamp; product; bid_price_1; bid_volume_1; bid_price_2;
  bid_volume_2; bid_price_3; bid_volume_3; ask_price_1; ask_volume_1;
  ask_price_2; ask_volume_2; ask_price_3; ask_volume_3; mid_price;
  profit_and_loss
- trades_round_{N}_day_{M}.csv -- semicolon-separated, one row per market trade.
  Columns: timestamp; buyer; seller; symbol; currency; price; quantity

Empty string means "level missing" (e.g. only 2 bid levels on this tick).

Snapshots are immutable. Call ``build_order_depth(product)`` to obtain a fresh
mutable ``OrderDepth`` per tick (the matching engine mutates book state as it
processes orders, so the cached snapshot must stay pristine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from backtester.datamodel import OrderDepth, Symbol, Trade

_PRICES_HEADER_COUNT = 17


@dataclass(frozen=True)
class PriceRow:
    """One row from the prices CSV. Immutable."""

    day: int
    timestamp: int
    product: Symbol
    bid_prices: Tuple[int, ...]
    bid_volumes: Tuple[int, ...]
    ask_prices: Tuple[int, ...]
    ask_volumes: Tuple[int, ...]
    mid_price: float
    profit_loss: float


@dataclass(frozen=True)
class TickSnapshot:
    """Everything the matching engine needs for one timestamp."""

    timestamp: int
    prices: Dict[Symbol, PriceRow] = field(default_factory=dict)
    market_trades: Dict[Symbol, List[Trade]] = field(default_factory=dict)

    def build_order_depth(self, product: Symbol) -> OrderDepth:
        """Return a FRESH OrderDepth for ``product`` (mutation-safe)."""
        od = OrderDepth()
        row = self.prices.get(product)
        if row is None:
            return od
        for price, volume in zip(row.bid_prices, row.bid_volumes):
            od.buy_orders[price] = volume
        for price, volume in zip(row.ask_prices, row.ask_volumes):
            od.sell_orders[price] = -volume
        return od


@dataclass(frozen=True)
class DayData:
    """Loaded day of historical data."""

    products: List[Symbol]
    snapshots: Dict[int, TickSnapshot]

    def timestamps(self) -> List[int]:
        return sorted(self.snapshots.keys())


def _parse_level_list(cells: Iterable[str]) -> Tuple[int, ...]:
    """Parse a sequence of cells, stopping at the first empty string."""
    out: List[int] = []
    for cell in cells:
        cell = cell.strip()
        if cell == "":
            break
        out.append(int(cell))
    return tuple(out)


def _read_prices(path: Path) -> Tuple[List[PriceRow], List[Symbol]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return [], []
    # Skip header.
    rows: List[PriceRow] = []
    products_seen: List[Symbol] = []
    seen_set: set[Symbol] = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split(";")
        if len(cols) < _PRICES_HEADER_COUNT:
            # pad with blanks so slicing works.
            cols = cols + [""] * (_PRICES_HEADER_COUNT - len(cols))
        bid_prices = _parse_level_list([cols[3], cols[5], cols[7]])
        bid_volumes = _parse_level_list([cols[4], cols[6], cols[8]])
        ask_prices = _parse_level_list([cols[9], cols[11], cols[13]])
        ask_volumes = _parse_level_list([cols[10], cols[12], cols[14]])
        # Trim to matched pairs so indices always align.
        n_bid = min(len(bid_prices), len(bid_volumes))
        n_ask = min(len(ask_prices), len(ask_volumes))
        row = PriceRow(
            day=int(cols[0]),
            timestamp=int(cols[1]),
            product=cols[2],
            bid_prices=bid_prices[:n_bid],
            bid_volumes=bid_volumes[:n_bid],
            ask_prices=ask_prices[:n_ask],
            ask_volumes=ask_volumes[:n_ask],
            mid_price=float(cols[15]) if cols[15].strip() else 0.0,
            profit_loss=float(cols[16]) if cols[16].strip() else 0.0,
        )
        rows.append(row)
        if row.product not in seen_set:
            seen_set.add(row.product)
            products_seen.append(row.product)
    return rows, sorted(products_seen)


def _read_trades(path: Path) -> List[Trade]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    trades: List[Trade] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split(";")
        if len(cols) < 7:
            continue
        try:
            trades.append(
                Trade(
                    symbol=cols[3],
                    price=int(float(cols[5])),
                    quantity=int(cols[6]),
                    buyer=cols[1] or None,
                    seller=cols[2] or None,
                    timestamp=int(cols[0]),
                )
            )
        except ValueError:
            # Skip malformed row; do not poison the whole day.
            continue
    return trades


def load_day(prices_path: Path, trades_path: Optional[Path] = None) -> DayData:
    """Load one day of Prosperity data.

    Args:
        prices_path: path to ``prices_round_N_day_M.csv``.
        trades_path: optional path to matching trades CSV.
    """
    prices_path = Path(prices_path)
    rows, products = _read_prices(prices_path)

    # Group prices by timestamp.
    prices_by_ts: Dict[int, Dict[Symbol, PriceRow]] = {}
    for row in rows:
        prices_by_ts.setdefault(row.timestamp, {})[row.product] = row

    trades_by_ts: Dict[int, Dict[Symbol, List[Trade]]] = {}
    if trades_path is not None and Path(trades_path).exists():
        for trade in _read_trades(Path(trades_path)):
            trades_by_ts.setdefault(trade.timestamp, {}).setdefault(trade.symbol, []).append(trade)

    snapshots: Dict[int, TickSnapshot] = {}
    for ts, product_map in prices_by_ts.items():
        snapshots[ts] = TickSnapshot(
            timestamp=ts,
            prices=product_map,
            market_trades=trades_by_ts.get(ts, {}),
        )

    return DayData(products=products, snapshots=snapshots)
