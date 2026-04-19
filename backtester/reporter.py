"""Reporter: JSON results + 4-panel matplotlib summary.

Panels:
  1. PnL curve (top-left)
  2. Position over time (top-right)
  3. Fill scatter (bottom-left) -- price vs. time, side-colored
  4. Drawdown curve (bottom-right)

Matplotlib is configured with a non-interactive backend. Library code writes
to disk and returns without leaking pyplot state.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtester.metrics import (
    compute_metrics,
    compute_pnl_curve,
    fill_quality_points,
)
from backtester.runner import RunResult


def _trade_dict(trade: Any) -> Dict[str, Any]:
    return {
        "symbol": trade.symbol,
        "price": trade.price,
        "quantity": trade.quantity,
        "buyer": trade.buyer,
        "seller": trade.seller,
        "timestamp": trade.timestamp,
    }


def _tick_dict(log: Any) -> Dict[str, Any]:
    return {
        "timestamp": log.timestamp,
        "duration_ms": round(log.duration_ms, 3),
        "trades": [_trade_dict(t) for t in log.trades],
        "position": log.position,
        "positions": dict(log.positions),
        "warnings": list(log.warnings),
        "rejections": list(log.rejections),
        "mid_prices": dict(log.mid_prices),
    }


def write_json(result: RunResult, path: Path) -> None:
    metrics = compute_metrics(result)
    doc: Dict[str, Any] = {
        "products": list(result.products),
        "final_positions": dict(result.final_positions),
        "final_trader_data": result.final_trader_data,
        "metrics": asdict(metrics),
        "tick_logs": [_tick_dict(log) for log in result.tick_logs],
    }
    # Round 2 fields are always present (zero / empty for Round 1) so that
    # downstream tooling can rely on a stable schema.
    doc["round2"] = {
        "total_fees_paid": result.total_fees_paid,
        "auction_outcomes": list(result.maf_auction_outcomes),
        "maf_bids_per_tick": list(result.maf_bids_per_tick),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))


def write_summary_plot(result: RunResult, path: Path) -> None:
    """Render the 4-panel matplotlib summary."""
    curve = compute_pnl_curve(result)
    timestamps = [p.timestamp for p in curve]
    pnls = [p.pnl for p in curve]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_pnl, ax_pos, ax_fills, ax_dd = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )

    # Panel 1: PnL curve
    ax_pnl.plot(timestamps, pnls, color="#1f77b4")
    ax_pnl.set_title("PnL (mark-to-mid)")
    ax_pnl.set_xlabel("timestamp")
    ax_pnl.set_ylabel("PnL")
    ax_pnl.grid(True, alpha=0.3)

    # Panel 2: Position per product
    for product in result.products:
        series_ts: List[int] = []
        series_pos: List[int] = []
        for log in result.tick_logs:
            series_ts.append(log.timestamp)
            series_pos.append(log.positions.get(product, 0))
        ax_pos.plot(series_ts, series_pos, label=product)
    ax_pos.set_title("Position")
    ax_pos.set_xlabel("timestamp")
    ax_pos.set_ylabel("position")
    ax_pos.axhline(0, color="gray", linewidth=0.8)
    if result.products:
        ax_pos.legend(loc="best", fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    # Panel 3: Fill scatter
    fills = fill_quality_points(result)
    for product, points in fills.items():
        buys = [(ts, px) for ts, px, s in points if s > 0]
        sells = [(ts, px) for ts, px, s in points if s < 0]
        if buys:
            ax_fills.scatter(
                [t for t, _ in buys],
                [p for _, p in buys],
                color="#2ca02c",
                marker="^",
                label=f"{product} buy",
                alpha=0.6,
                s=30,
            )
        if sells:
            ax_fills.scatter(
                [t for t, _ in sells],
                [p for _, p in sells],
                color="#d62728",
                marker="v",
                label=f"{product} sell",
                alpha=0.6,
                s=30,
            )
    ax_fills.set_title("Fills (price vs. time)")
    ax_fills.set_xlabel("timestamp")
    ax_fills.set_ylabel("price")
    if fills:
        ax_fills.legend(loc="best", fontsize=8)
    ax_fills.grid(True, alpha=0.3)

    # Panel 4: Drawdown
    peak = float("-inf") if not pnls else pnls[0]
    dd_series: List[float] = []
    for pnl in pnls:
        peak = max(peak, pnl)
        dd_series.append(peak - pnl)
    ax_dd.plot(timestamps, dd_series, color="#d62728")
    ax_dd.set_title("Drawdown")
    ax_dd.set_xlabel("timestamp")
    ax_dd.set_ylabel("peak - pnl")
    ax_dd.grid(True, alpha=0.3)

    fig.suptitle(f"Prosperity Backtest Summary ({', '.join(result.products)})")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
