"""Metrics: PnL curve, drawdown, Sharpe, fill quality.

PnL convention (mark-to-mid, sum over products):

    For a buy of q at price p: cash -= q * p
    For a sell of q at price p: cash += q * p
    inventory_value = sum_p (position[p] * mid[p])
    pnl(t) = cash(t) + inventory_value(t)

Trades in TickLog follow the matching-engine convention:
  - buyer == "SUBMISSION" -> user bought (position += qty)
  - seller == "SUBMISSION" -> user sold (position -= qty)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from backtester.runner import RunResult


@dataclass(frozen=True)
class PnLPoint:
    timestamp: int
    pnl: float


@dataclass(frozen=True)
class Metrics:
    final_pnl: float
    max_drawdown: float
    num_trades: int
    sharpe: Optional[float]
    avg_position: float
    max_position_abs: int


def compute_pnl_curve(result: RunResult) -> List[PnLPoint]:
    """Compute mark-to-mid PnL at each tick."""
    cash = 0.0
    curve: List[PnLPoint] = []
    for log in result.tick_logs:
        for trade in log.trades:
            if trade.buyer == "SUBMISSION":
                cash -= trade.price * trade.quantity
            elif trade.seller == "SUBMISSION":
                cash += trade.price * trade.quantity
        inventory_value = 0.0
        for product, pos in log.positions.items():
            mid = log.mid_prices.get(product)
            if mid is None:
                continue
            inventory_value += pos * mid
        curve.append(PnLPoint(timestamp=log.timestamp, pnl=cash + inventory_value))
    return curve


def compute_drawdown(curve: List[PnLPoint]) -> float:
    """Maximum peak-to-trough drawdown of the PnL curve."""
    if not curve:
        return 0.0
    peak = curve[0].pnl
    worst = 0.0
    for point in curve:
        peak = max(peak, point.pnl)
        dd = peak - point.pnl
        if dd > worst:
            worst = dd
    return worst


def compute_sharpe(curve: List[PnLPoint]) -> float:
    """Simple Sharpe: mean(dpnl) / std(dpnl). Tick units; no annualization."""
    if len(curve) < 2:
        return 0.0
    diffs = [curve[i + 1].pnl - curve[i].pnl for i in range(len(curve) - 1)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / n
    if var == 0:
        return 0.0
    return mean / math.sqrt(var)


def compute_metrics(result: RunResult) -> Metrics:
    curve = compute_pnl_curve(result)
    num_trades = sum(len(log.trades) for log in result.tick_logs)
    if result.tick_logs:
        avg_position = sum(
            abs(log.position) for log in result.tick_logs
        ) / len(result.tick_logs)
        max_position_abs = max(abs(log.position) for log in result.tick_logs)
    else:
        avg_position = 0.0
        max_position_abs = 0
    return Metrics(
        final_pnl=curve[-1].pnl if curve else 0.0,
        max_drawdown=compute_drawdown(curve),
        num_trades=num_trades,
        sharpe=compute_sharpe(curve),
        avg_position=avg_position,
        max_position_abs=max_position_abs,
    )


def fill_quality_points(result: RunResult) -> Dict[str, List[tuple[int, int, int]]]:
    """Per-product list of (ts, price, sign). +1 = user buy, -1 = user sell."""
    out: Dict[str, List[tuple[int, int, int]]] = {}
    for log in result.tick_logs:
        for trade in log.trades:
            sign = 1 if trade.buyer == "SUBMISSION" else -1
            out.setdefault(trade.symbol, []).append((log.timestamp, trade.price, sign))
    return out
