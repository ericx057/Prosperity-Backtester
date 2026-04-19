"""CLI entry point for a single-day backtest.

Usage:
    python backtest.py --data PATH --trader PATH [--config PATH] [--out DIR]

Exit code 0 on success. Prints the final PnL and rejection count to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from backtester.data_loader import load_day
from backtester.reporter import write_json, write_summary_plot
from backtester.round2 import Round2Config, load_round2_config_from_yaml
from backtester.runner import BacktestConfig, run_backtest


DEFAULT_POSITION_LIMITS: Dict[str, int] = {
    "RAINFOREST_RESIN": 50,
    "RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}


def load_trader(path: str) -> Any:
    """Dynamically import ``Trader`` class from the given file path."""
    abs_path = Path(path).resolve()
    # Ensure the trader's parent dir is importable (for ``from datamodel import ...``).
    parent = str(abs_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # Also make backtester.datamodel importable as "datamodel" for
    # trader submissions.
    _install_datamodel_shim()
    spec = importlib.util.spec_from_file_location(abs_path.stem, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import trader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Trader"):
        raise AttributeError(
            f"Module {path} does not define a Trader class"
        )
    return module.Trader()


def _install_datamodel_shim() -> None:
    """Make ``from datamodel import ...`` work inside a trader file.

    The official Prosperity trader code does ``from datamodel import Order, ...``
    because the live environment places ``datamodel`` at the top level. We
    alias our ``backtester.datamodel`` as ``datamodel`` so traders work
    unchanged.
    """
    if "datamodel" in sys.modules:
        return
    from backtester import datamodel as _dm
    sys.modules["datamodel"] = _dm


def load_run_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def fidelity_lock_hash() -> str:
    """Hash of matching_engine.py for the fidelity lock."""
    engine = Path("backtester/matching_engine.py").resolve()
    if not engine.exists():
        return ""
    return hashlib.sha256(engine.read_bytes()).hexdigest()


def check_fidelity_lock(lock_path: Path) -> bool:
    """Return True if lock matches current matching engine hash."""
    if not lock_path.exists():
        return False
    return lock_path.read_text().strip() == fidelity_lock_hash()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Prosperity backtest")
    parser.add_argument("--data", required=True, help="prices CSV path")
    parser.add_argument("--trades", default=None, help="optional trades CSV path")
    parser.add_argument("--trader", required=True, help="trader .py path")
    parser.add_argument("--config", default=None, help="optional run config YAML")
    parser.add_argument("--out", default="out/backtest", help="output directory")
    parser.add_argument(
        "--round2-config",
        default=None,
        help="optional YAML configuring the Round 2 MAF auction",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    lock_path = Path(".fidelity_lock")
    if not check_fidelity_lock(lock_path):
        print(
            "WARNING: fidelity suite has not been re-run for the current "
            "matching_engine.py. Run `make fidelity` and commit the updated "
            ".fidelity_lock before trusting sweep or backtest results.",
            file=sys.stderr,
        )

    prices_path = Path(args.data)
    trades_path = Path(args.trades) if args.trades else None
    # Auto-detect trades CSV next to prices CSV if not specified.
    if trades_path is None:
        guess = prices_path.with_name(
            prices_path.name.replace("prices", "trades")
        )
        if guess.exists() and guess != prices_path:
            trades_path = guess
    data = load_day(prices_path, trades_path)

    trader = load_trader(args.trader)

    run_cfg = load_run_config(args.config)
    position_limits = dict(DEFAULT_POSITION_LIMITS)
    position_limits.update(run_cfg.get("position_limits", {}))
    # Back-fill any products seen in data with a sane default.
    for product in data.products:
        if product not in position_limits:
            position_limits[product] = 50

    round2_cfg: Optional[Round2Config] = None
    if args.round2_config:
        round2_cfg = load_round2_config_from_yaml(args.round2_config)
    elif "round2" in run_cfg:
        # Allow embedding the Round 2 block inside the main run config.
        from backtester.round2 import round2_config_from_dict
        round2_cfg = round2_config_from_dict(run_cfg["round2"])

    config = BacktestConfig(
        position_limits=position_limits,
        timeout_ms=int(run_cfg.get("timeout_ms", 900)),
        yellow_threshold_ms=int(run_cfg.get("yellow_threshold_ms", 500)),
        seed=run_cfg.get("seed"),
        round2=round2_cfg,
    )

    result = run_backtest(trader, data, config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    plot_path = out_dir / "summary.png"
    write_json(result, json_path)
    write_summary_plot(result, plot_path)

    total_trades = sum(len(log.trades) for log in result.tick_logs)
    total_rejections = sum(len(log.rejections) for log in result.tick_logs)
    total_warnings = sum(len(log.warnings) for log in result.tick_logs)

    from backtester.metrics import compute_metrics

    metrics = compute_metrics(result)
    print(f"products: {result.products}")
    print(f"final PnL: {metrics.final_pnl:.2f}")
    print(f"max drawdown: {metrics.max_drawdown:.2f}")
    print(f"trades: {total_trades}  rejections: {total_rejections}  warnings: {total_warnings}")
    print(f"final positions: {result.final_positions}")
    if round2_cfg is not None and round2_cfg.enabled:
        num_wins = sum(
            1
            for tick in result.maf_auction_outcomes
            for o in tick["outcomes"].values()
            if o["won"]
        )
        num_events = sum(
            len(tick["outcomes"]) for tick in result.maf_auction_outcomes
        )
        print(
            f"round2: fees_paid={result.total_fees_paid:.4f}  "
            f"wins={num_wins}/{num_events}"
        )
    print(f"json: {json_path}")
    print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
