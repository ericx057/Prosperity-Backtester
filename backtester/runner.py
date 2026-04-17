"""Backtest runner - tick loop, trader invocation, fidelity enforcement.

Responsibilities:
- For each timestamp: rebuild state, call ``trader.run(state)``, enforce
  the timeout and JSON size budgets, feed orders to the matching engine,
  and record per-tick logs.

Does NOT know anything about the matching algorithm itself. All order-match
logic lives in ``backtester.matching_engine``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from backtester.data_loader import DayData
from backtester.datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Symbol,
    Trade,
    TradingState,
)
from backtester.matching_engine import MatchingEngine

logger = logging.getLogger(__name__)

DEFAULT_TRADER_DATA_MAX_BYTES = 1 * 1024 * 1024
DEFAULT_TIMEOUT_MS = 900
DEFAULT_YELLOW_MS = 500


class TraderProtocol(Protocol):
    """Any object with a ``run(state) -> (orders, conversions, traderData)`` method."""

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        ...


@dataclass(frozen=True)
class BacktestConfig:
    position_limits: Dict[Symbol, int]
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    yellow_threshold_ms: int = DEFAULT_YELLOW_MS
    trader_data_max_bytes: int = DEFAULT_TRADER_DATA_MAX_BYTES
    seed: Optional[int] = None


@dataclass
class TickLog:
    timestamp: int
    duration_ms: float
    trades: List[Trade]
    position: int
    positions: Dict[Symbol, int]
    warnings: List[str]
    rejections: List[str]
    mid_prices: Dict[Symbol, float]


@dataclass
class RunResult:
    tick_logs: List[TickLog]
    final_positions: Dict[Symbol, int]
    final_trader_data: str
    products: List[Symbol]


def _build_state(
    *,
    timestamp: int,
    data: DayData,
    trader_data: str,
    position: Dict[Symbol, int],
    own_trades: Dict[Symbol, List[Trade]],
    prev_market_trades: Dict[Symbol, List[Trade]],
) -> TradingState:
    snapshot = data.snapshots[timestamp]
    order_depths: Dict[Symbol, OrderDepth] = {}
    listings: Dict[Symbol, Listing] = {}
    for product in data.products:
        order_depths[product] = snapshot.build_order_depth(product)
        listings[product] = Listing(product, product, 1)
    observations = Observation({}, {})
    return TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=prev_market_trades,
        position=position,
        observations=observations,
    )


def _serialize_trader_data(
    raw: Any, max_bytes: int
) -> tuple[str, Optional[str]]:
    """Round-trip traderData through json.dumps. Return (safe_str, warning)."""
    if raw is None:
        return "", None
    if isinstance(raw, str):
        payload = raw
    else:
        try:
            payload = json.dumps(raw)
        except (TypeError, ValueError) as exc:
            return "", f"traderData not JSON-serializable: {exc}"
    try:
        size = len(payload.encode("utf-8"))
    except UnicodeEncodeError as exc:
        return "", f"traderData encoding error: {exc}"
    if size > max_bytes:
        return "", f"traderData exceeds size cap ({size} > {max_bytes} bytes)"
    return payload, None


def _safe_run(
    trader: TraderProtocol,
    state: TradingState,
    config: BacktestConfig,
) -> tuple[Optional[Dict[Symbol, List[Order]]], int, Any, float, List[str]]:
    warnings: List[str] = []
    start = time.perf_counter()
    try:
        orders, conversions, raw_trader_data = trader.run(state)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        warnings.append(f"trader.run raised exception: {exc!r}; orders dropped")
        return None, 0, "", duration_ms, warnings
    duration_ms = (time.perf_counter() - start) * 1000
    if duration_ms > config.timeout_ms:
        warnings.append(
            f"trader.run timeout: {duration_ms:.1f}ms > {config.timeout_ms}ms; orders dropped"
        )
        return None, 0, "", duration_ms, warnings
    if duration_ms > config.yellow_threshold_ms:
        warnings.append(
            f"trader.run slow: {duration_ms:.1f}ms > {config.yellow_threshold_ms}ms yellow threshold"
        )
    return orders, conversions, raw_trader_data, duration_ms, warnings


def run_backtest(
    trader: TraderProtocol,
    data: DayData,
    config: BacktestConfig,
    progress: Optional[Callable[[int, int], None]] = None,
) -> RunResult:
    """Execute a full day backtest."""
    if config.seed is not None:
        import random
        random.seed(config.seed)
        try:
            import numpy as np
            np.random.seed(config.seed)
        except ImportError:
            pass

    engine = MatchingEngine(position_limits=config.position_limits)
    timestamps = data.timestamps()

    position: Dict[Symbol, int] = {product: 0 for product in data.products}
    own_trades: Dict[Symbol, List[Trade]] = {}
    market_trades_prev: Dict[Symbol, List[Trade]] = {}
    trader_data: str = ""

    tick_logs: List[TickLog] = []

    for i, ts in enumerate(timestamps):
        state = _build_state(
            timestamp=ts,
            data=data,
            trader_data=trader_data,
            position=dict(position),
            own_trades=own_trades,
            prev_market_trades=market_trades_prev,
        )

        orders, _conversions, raw_trader_data, duration_ms, warnings = _safe_run(
            trader, state, config
        )

        tick_trades: List[Trade] = []
        tick_rejections: List[str] = []
        tick_own_trades: Dict[Symbol, List[Trade]] = {}

        if orders is not None and not isinstance(orders, dict):
            warnings.append(
                f"trader.run returned non-dict orders ({type(orders).__name__}); dropped"
            )
            orders = None

        if orders is not None:
            snapshot = data.snapshots[ts]
            for product in data.products:
                product_orders = orders.get(product, [])
                if not product_orders:
                    continue
                book = snapshot.build_order_depth(product)
                market_trade_list = snapshot.market_trades.get(product, [])
                result = engine.match(
                    symbol=product,
                    user_orders=product_orders,
                    book=book,
                    position=position.get(product, 0),
                    market_trades=market_trade_list,
                    timestamp=ts,
                )
                position[product] = result.new_position
                tick_trades.extend(result.trades)
                tick_rejections.extend(result.rejections)
                if result.trades:
                    tick_own_trades[product] = result.trades

            serialized, warn = _serialize_trader_data(
                raw_trader_data, config.trader_data_max_bytes
            )
            if warn is not None:
                warnings.append(warn)
                trader_data = ""
            else:
                trader_data = serialized

        own_trades = tick_own_trades
        market_trades_prev = dict(data.snapshots[ts].market_trades)

        tick_logs.append(
            TickLog(
                timestamp=ts,
                duration_ms=duration_ms,
                trades=tick_trades,
                position=position.get(data.products[0], 0) if data.products else 0,
                positions=dict(position),
                warnings=warnings,
                rejections=tick_rejections,
                mid_prices={
                    p: data.snapshots[ts].prices[p].mid_price
                    for p in data.products
                    if p in data.snapshots[ts].prices
                },
            )
        )

        if progress is not None:
            progress(i + 1, len(timestamps))

    return RunResult(
        tick_logs=tick_logs,
        final_positions=dict(position),
        final_trader_data=trader_data,
        products=list(data.products),
    )
