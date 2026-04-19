"""Parameter sweep engine.

Design rules (PRD section 8):
- Multiprocessing via ``concurrent.futures.ProcessPoolExecutor``.
- NO early stopping; sweep the entire landscape for plateau analysis.
- Structured CSV output, one row per parameter combination.
- Deterministic seeding.
- Emit a heatmap PNG per pair of swept dimensions.

The sweeper does not pick a "best" config. Plateau identification is an
offline step (see README: ±10% / 80%-of-plateau rule).
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from backtester.data_loader import DayData
from backtester.metrics import compute_metrics
from backtester.round2 import Round2Config, round2_config_from_dict
from backtester.runner import BacktestConfig, run_backtest


@dataclass(frozen=True)
class SweepParam:
    name: str
    values: List[Any]


@dataclass(frozen=True)
class TraderFileFactory:
    """Pickleable factory that loads a Trader class from a .py path on call.

    Required because ``ProcessPoolExecutor`` uses ``spawn`` on macOS — child
    processes cannot see closures or dynamically-registered modules from the
    parent. This factory is a plain dataclass (pickleable) and does the
    ``importlib`` work inside the worker.
    """

    path: str
    class_name: str = "Trader"
    base_kwargs: Dict[str, Any] = field(default_factory=dict)
    module_name: str = "_sweep_trader_module"

    def __call__(self, **overrides: Any) -> Any:
        if "datamodel" not in sys.modules:
            from backtester import datamodel as _dm

            sys.modules["datamodel"] = _dm
        mod = sys.modules.get(self.module_name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(
                self.module_name, self.path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load trader module from {self.path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[self.module_name] = mod
            spec.loader.exec_module(mod)
        cls = getattr(mod, self.class_name)
        merged = {**self.base_kwargs, **overrides}
        return cls(**merged)


@dataclass
class SweepConfig:
    trader_factory: Callable[..., Any]
    params: List[SweepParam]
    position_limits: Dict[str, int]
    workers: int = 1
    seed: Optional[int] = None
    timeout_ms: int = 900
    # Optional Round 2 base config. Swept Round 2 dimensions (e.g.
    # competition_threshold) are merged on top per combo via params whose
    # name is prefixed with ``round2.``.
    round2: Optional[Round2Config] = None


def cartesian_combos(params: List[SweepParam]) -> List[Dict[str, Any]]:
    """Enumerate the full cartesian product of parameter values."""
    if not params:
        return [{}]
    names = [p.name for p in params]
    value_lists = [p.values for p in params]
    combos: List[Dict[str, Any]] = []
    for tup in itertools.product(*value_lists):
        combos.append({names[i]: tup[i] for i in range(len(names))})
    return combos


_ROUND2_PARAM_PREFIX = "round2."


def _split_combo(combo: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a combo into (trader_kwargs, round2_overrides).

    Any param name starting with ``round2.`` is routed into the Round 2
    config for that run. Everything else is passed to the trader factory.
    """
    trader_kwargs: Dict[str, Any] = {}
    round2_overrides: Dict[str, Any] = {}
    for key, val in combo.items():
        if key.startswith(_ROUND2_PARAM_PREFIX):
            round2_overrides[key[len(_ROUND2_PARAM_PREFIX):]] = val
        else:
            trader_kwargs[key] = val
    return trader_kwargs, round2_overrides


def _apply_round2_overrides(
    base: Optional[Round2Config], overrides: Dict[str, Any]
) -> Optional[Round2Config]:
    """Return a new Round2Config with the given field overrides applied.

    If both ``base`` is ``None`` and ``overrides`` is empty, returns ``None``.
    """
    if not overrides:
        return base
    if base is None:
        # User is sweeping Round 2 dims without providing a base - start from
        # defaults and then apply overrides.
        return round2_config_from_dict(overrides)
    # Merge: base first, then overrides win.
    merged: Dict[str, Any] = dict(base.__dict__)
    merged.update(overrides)
    return round2_config_from_dict(merged)


def _run_one(args: tuple) -> Dict[str, Any]:
    """Worker entry point. Must be module-level for pickling."""
    (
        combo,
        trader_factory,
        data,
        position_limits,
        seed,
        timeout_ms,
        base_round2,
    ) = args
    trader_kwargs, round2_overrides = _split_combo(combo)
    trader = trader_factory(**trader_kwargs)
    round2_cfg = _apply_round2_overrides(base_round2, round2_overrides)
    config = BacktestConfig(
        position_limits=position_limits,
        seed=seed,
        timeout_ms=timeout_ms,
        round2=round2_cfg,
    )
    result = run_backtest(trader, data, config)
    metrics = compute_metrics(result)
    row: Dict[str, Any] = dict(combo)
    row.update(
        final_pnl=metrics.final_pnl,
        max_drawdown=metrics.max_drawdown,
        sharpe=metrics.sharpe if metrics.sharpe is not None else 0.0,
        num_trades=metrics.num_trades,
        avg_position=metrics.avg_position,
        max_position_abs=metrics.max_position_abs,
        total_fees_paid=result.total_fees_paid,
        net_pnl=metrics.final_pnl - result.total_fees_paid,
    )
    return row


def run_sweep(cfg: SweepConfig, data: DayData) -> List[Dict[str, Any]]:
    """Execute the full cartesian sweep. Returns one row per combo."""
    combos = cartesian_combos(cfg.params)
    tasks = [
        (
            combo,
            cfg.trader_factory,
            data,
            cfg.position_limits,
            cfg.seed,
            cfg.timeout_ms,
            cfg.round2,
        )
        for combo in combos
    ]

    if cfg.workers <= 1:
        return [_run_one(task) for task in tasks]

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def write_sweep_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        Path(path).write_text("")
        return
    # Stable column order: param columns alphabetically, then metric columns.
    metric_cols = [
        "final_pnl",
        "max_drawdown",
        "sharpe",
        "num_trades",
        "avg_position",
        "max_position_abs",
        "total_fees_paid",
        "net_pnl",
    ]
    all_keys = set().union(*(r.keys() for r in rows))
    param_cols = sorted(all_keys - set(metric_cols))
    fieldnames = param_cols + [c for c in metric_cols if c in all_keys]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_heatmap(
    rows: List[Dict[str, Any]],
    x_param: str,
    y_param: str,
    metric: str,
    path: Path,
) -> None:
    """Render a 2D heatmap over (x_param, y_param). Values averaged if duplicates."""
    x_values = sorted({row[x_param] for row in rows})
    y_values = sorted({row[y_param] for row in rows})
    x_idx = {v: i for i, v in enumerate(x_values)}
    y_idx = {v: i for i, v in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan)
    counts = np.zeros_like(grid)
    for row in rows:
        xi = x_idx[row[x_param]]
        yi = y_idx[row[y_param]]
        val = float(row[metric])
        if np.isnan(grid[yi, xi]):
            grid[yi, xi] = 0.0
        grid[yi, xi] += val
        counts[yi, xi] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest"
    )
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(v) for v in y_values])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{metric} over ({x_param}, {y_param})")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_all_heatmaps(
    rows: List[Dict[str, Any]],
    param_names: List[str],
    metric: str,
    out_dir: Path,
) -> List[Path]:
    """For each pair of swept dims, emit a heatmap. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            x, y = param_names[i], param_names[j]
            path = out_dir / f"heatmap_{metric}_{x}_vs_{y}.png"
            write_heatmap(rows, x, y, metric, path)
            paths.append(path)
    return paths
