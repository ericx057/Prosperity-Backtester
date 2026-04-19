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
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Union

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
from backtester.round2 import (
    MAFAuctionResult,
    Round2Config,
    _SCALAR_PRODUCT_KEY,
    is_scalar_result,
    resolve_maf_auction,
)

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
    # Opt-in Round 2 auction config. ``None`` = pre-Round-2 behavior.
    round2: Optional[Round2Config] = None


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
    # Round 2 accumulators. Empty / zero for Round 1 runs so that pre-Round-2
    # callers continue to see meaningful defaults.
    total_fees_paid: float = 0.0
    maf_auction_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    maf_bids_per_tick: List[Dict[str, Any]] = field(default_factory=list)


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


def _extract_maf(
    trader: Any,
    state: TradingState,
    raw_trader_data: Any,
    cfg: Round2Config,
) -> Optional[Union[float, Dict[Symbol, float]]]:
    """Look up the trader's MAF declaration for this tick.

    Resolution order (all names configurable via ``Round2Config``):

    1. ``getattr(trader, cfg.maf_method_name)`` - call with ``state``.
       Return value must be a ``float``, ``int``, or ``Mapping[str, float]``.
    2. ``getattr(trader, cfg.maf_attribute_name)`` - a scalar or mapping.
    3. ``raw_trader_data[cfg.maf_field_name]`` - if traderData is a dict
       (or a JSON-encoded dict), pull the named field.

    Returns ``None`` if no MAF declaration can be found.
    """
    # Path 1: configured method.
    method = getattr(trader, cfg.maf_method_name, None)
    if callable(method):
        try:
            raw = method(state)
        except Exception:
            # Never let MAF-lookup crash derail the tick loop.
            logger.warning("MAF method %r raised; treating as no declaration.", cfg.maf_method_name)
            return None
        return _coerce_maf(raw)

    # Path 2: configured attribute.
    attr_val = getattr(trader, cfg.maf_attribute_name, None)
    if attr_val is not None:
        return _coerce_maf(attr_val)

    # Path 3: field on traderData (expect a dict or JSON-decodable string).
    data: Any = raw_trader_data
    if isinstance(data, str) and data:
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            data = None
    if isinstance(data, Mapping) and cfg.maf_field_name in data:
        return _coerce_maf(data[cfg.maf_field_name])

    return None


def _coerce_maf(
    raw: Any,
) -> Optional[Union[float, Dict[Symbol, float]]]:
    """Coerce a raw MAF value into a scalar or per-product mapping, or None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, Mapping):
        out: Dict[Symbol, float] = {}
        for key, val in raw.items():
            if not isinstance(key, str):
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[key] = float(val)
        return out if out else None
    return None


def _auction_result_to_dict(res: MAFAuctionResult) -> Dict[str, Any]:
    return {
        "won": res.won,
        "fee_paid": res.fee_paid,
        "volume_multiplier": res.volume_multiplier,
        "declared_maf": res.declared_maf,
    }


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

    # Round 2 state. Only live when round2 is configured AND enabled.
    round2_active: bool = (
        config.round2 is not None and config.round2.enabled
    )
    auction_rng: Optional[random.Random] = None
    if round2_active:
        assert config.round2 is not None  # for type checker
        auction_seed = config.round2.auction_seed
        if auction_seed is None:
            auction_seed = config.seed
        auction_rng = random.Random(auction_seed)
    total_fees_paid: float = 0.0
    auction_outcomes: List[Dict[str, Any]] = []
    maf_bids: List[Dict[str, Any]] = []
    # Cached "once" outcome. Keyed by product symbol (or scalar sentinel).
    cached_outcomes: Optional[Dict[str, MAFAuctionResult]] = None

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

        # ---- Round 2: resolve the auction BEFORE matching ----
        tick_auction: Dict[str, MAFAuctionResult] = {}
        if round2_active and orders is not None:
            assert config.round2 is not None
            assert auction_rng is not None
            r2cfg = config.round2
            if r2cfg.auction_frequency == "once" and cached_outcomes is not None:
                tick_auction = cached_outcomes
            else:
                trader_maf = _extract_maf(trader, state, raw_trader_data, r2cfg)
                tick_auction = resolve_maf_auction(
                    trader_maf, r2cfg, auction_rng
                )
                if r2cfg.auction_frequency == "once":
                    cached_outcomes = tick_auction
                maf_bids.append({
                    "timestamp": ts,
                    "maf": _maf_to_json(trader_maf),
                })

            for product_key, outcome in tick_auction.items():
                if outcome.won:
                    total_fees_paid += outcome.fee_paid
            auction_outcomes.append(
                {
                    "timestamp": ts,
                    "outcomes": {
                        k: _auction_result_to_dict(v)
                        for k, v in tick_auction.items()
                    },
                }
            )

        if orders is not None:
            snapshot = data.snapshots[ts]
            for product in data.products:
                product_orders = orders.get(product, [])
                if not product_orders:
                    continue
                book = snapshot.build_order_depth(product)
                market_trade_list = snapshot.market_trades.get(product, [])

                vol_mult = _volume_multiplier_for(
                    product=product,
                    tick_auction=tick_auction,
                    round2_active=round2_active,
                )
                result = engine.match(
                    symbol=product,
                    user_orders=product_orders,
                    book=book,
                    position=position.get(product, 0),
                    market_trades=market_trade_list,
                    timestamp=ts,
                    volume_multiplier=vol_mult,
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
        total_fees_paid=total_fees_paid,
        maf_auction_outcomes=auction_outcomes,
        maf_bids_per_tick=maf_bids,
    )


def _volume_multiplier_for(
    *,
    product: Symbol,
    tick_auction: Mapping[str, MAFAuctionResult],
    round2_active: bool,
) -> float:
    """Return the volume multiplier to apply to ``product``'s book this tick.

    - If Round 2 is inactive OR there are no auction results, return 1.0.
    - If the auction was on a scalar (round-level) MAF, the same multiplier
      applies to every product.
    - Otherwise, look up the per-product outcome. Missing outcome => 1.0.
    """
    if not round2_active or not tick_auction:
        return 1.0
    if is_scalar_result(tick_auction):
        return tick_auction[_SCALAR_PRODUCT_KEY].volume_multiplier
    outcome = tick_auction.get(product)
    if outcome is None:
        return 1.0
    return outcome.volume_multiplier


def _maf_to_json(
    raw: Optional[Union[float, Dict[Symbol, float]]],
) -> Any:
    """Return a JSON-safe representation of a MAF declaration."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    return float(raw)
