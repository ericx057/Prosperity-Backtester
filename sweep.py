"""CLI entry point for a parameter sweep.

Usage:
    python sweep.py --data PATH --trader PATH --sweep-config PATH [--workers N] [--out DIR]

Produces:
  - sweep_results.csv in the output dir.
  - heatmap PNGs per pair of swept dimensions.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from backtester.data_loader import load_day
from backtester.sweeper import (
    SweepConfig,
    SweepParam,
    TraderFileFactory,
    run_sweep,
    write_all_heatmaps,
    write_sweep_csv,
)


def _install_datamodel_shim() -> None:
    if "datamodel" in sys.modules:
        return
    from backtester import datamodel as _dm
    sys.modules["datamodel"] = _dm


def _resolve_trader_file(spec: str, search_paths: list[Path]) -> tuple[Path, str]:
    """Resolve ``module.Class`` or ``Class`` to ``(file_path, class_name)``.

    Returns an absolute path to the trader ``.py`` and the class name. We
    refuse to resolve to an already-imported module because the child worker
    processes (spawn start method) cannot see the parent's ``sys.modules``.
    A concrete file path is the only pickleable handoff.
    """
    if "." in spec:
        module_name, class_name = spec.rsplit(".", 1)
    else:
        module_name, class_name = "trader", spec
    for sp in search_paths:
        candidate = sp / f"{module_name}.py"
        if candidate.exists():
            return candidate.resolve(), class_name
    raise FileNotFoundError(
        f"cannot find {module_name}.py on any of: {[str(p) for p in search_paths]}"
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Prosperity parameter sweep")
    parser.add_argument("--data", required=True)
    parser.add_argument("--trades", default=None)
    parser.add_argument(
        "--trader", default=None, help="trader .py path (inferred from sweep config if omitted)"
    )
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default="out/sweep")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    with open(args.sweep_config) as f:
        cfg = yaml.safe_load(f)

    search_paths = [Path.cwd()]
    if args.trader:
        trader_path = Path(args.trader).resolve()
        search_paths.insert(0, trader_path.parent)

    trader_kwargs = cfg.get("trader_kwargs", {}) or {}

    if args.trader and Path(args.trader).exists():
        # Explicit --trader path wins over the sweep config's trader_class name.
        spec = cfg.get("trader_class", "Trader")
        class_name = spec.rsplit(".", 1)[-1]
        trader_file_path = Path(args.trader).resolve()
    else:
        trader_file_path, class_name = _resolve_trader_file(
            cfg["trader_class"], search_paths
        )

    factory = TraderFileFactory(
        path=str(trader_file_path),
        class_name=class_name,
        base_kwargs=trader_kwargs,
    )
    _install_datamodel_shim()

    params = [
        SweepParam(name=name, values=list(values))
        for name, values in (cfg.get("params") or {}).items()
    ]

    position_limits: Dict[str, int] = cfg.get("position_limits", {})

    data = load_day(Path(args.data), Path(args.trades) if args.trades else None)

    workers = args.workers if args.workers is not None else int(cfg.get("workers", 1))

    sweep_cfg = SweepConfig(
        trader_factory=factory,
        params=params,
        position_limits=position_limits,
        workers=workers,
        seed=cfg.get("seed"),
    )

    rows = run_sweep(sweep_cfg, data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_results.csv"
    write_sweep_csv(rows, csv_path)

    metric = cfg.get("metric", "final_pnl")
    heatmap_paths = write_all_heatmaps(
        rows, [p.name for p in params], metric, out_dir
    )

    print(f"combos: {len(rows)}")
    print(f"csv: {csv_path}")
    for hp in heatmap_paths:
        print(f"heatmap: {hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
