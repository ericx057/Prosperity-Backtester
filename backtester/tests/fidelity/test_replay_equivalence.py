"""Replay-equivalence harness (PRD section 7.1).

The harness itself is fully functional. What is NOT yet supplied is the
``replay_cases`` list of live-log expected outputs — those must be populated
from submitted runs as described in the README.

To add a new replay case:
    1. Run a trader on the live Prosperity platform.
    2. Download the prices + trades CSVs for that run, plus the activity log
       showing per-tick PnL and position.
    3. Drop the CSVs into ``backtester/tests/fidelity/replay_fixtures/{name}/``
       and create an ``expected.json`` with the schema:

         {
           "trader_module": "example_trader",
           "trader_class": "Trader",
           "prices_csv": "prices_round_1_day_0.csv",
           "trades_csv": "trades_round_1_day_0.csv",
           "position_limits": {"RESIN": 50, ...},
           "final_pnl_live": 12345.67,
           "tolerance_pnl_pct": 0.02,
           "position_trajectory": [
             {"timestamp": 0, "positions": {"RESIN": 0}},
             ...
           ]
         }

    4. Add the directory name to ``REPLAY_CASES`` below (or rely on autoload).
    5. Run ``pytest -m replay``.

The invariants this test asserts:
  - |final_pnl_backtest - final_pnl_live| / |final_pnl_live| <= tolerance
    (default 2%, PRD section 3).
  - For every tick in the expected trajectory, the backtested position
    matches within +-1 unit (PRD section 3).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest

from backtester.data_loader import load_day
from backtester.runner import BacktestConfig, run_backtest

pytestmark = [pytest.mark.fidelity, pytest.mark.replay]

REPLAY_DIR = Path(__file__).parent / "replay_fixtures"
POSITION_TOLERANCE = 1
DEFAULT_PNL_TOLERANCE_PCT = 0.02


@dataclass(frozen=True)
class ReplayCase:
    directory: Path
    spec: dict

    @property
    def name(self) -> str:
        return self.directory.name


def _discover_cases() -> List[ReplayCase]:
    if not REPLAY_DIR.exists():
        return []
    cases: List[ReplayCase] = []
    for child in sorted(REPLAY_DIR.iterdir()):
        if not child.is_dir():
            continue
        spec_path = child / "expected.json"
        if not spec_path.exists():
            continue
        try:
            spec = json.loads(spec_path.read_text())
        except json.JSONDecodeError:
            continue
        cases.append(ReplayCase(directory=child, spec=spec))
    return cases


def _load_trader(module: str, cls: str, search_path: Path):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
    # Ensure `datamodel` shim works for user traders.
    from backtester import datamodel as _dm
    sys.modules.setdefault("datamodel", _dm)
    try:
        mod = importlib.import_module(module)
    except ImportError:
        candidate = search_path / f"{module}.py"
        spec = importlib.util.spec_from_file_location(module, candidate)
        assert spec and spec.loader, f"Cannot load {candidate}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return getattr(mod, cls)()


CASES = _discover_cases()


@pytest.mark.skipif(not CASES, reason="no replay fixtures present — see test docstring to populate")
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_replay_equivalence(case: ReplayCase) -> None:
    spec = case.spec
    trader_search = case.directory
    trader = _load_trader(spec["trader_module"], spec["trader_class"], trader_search)

    prices_path = case.directory / spec["prices_csv"]
    trades_path = case.directory / spec["trades_csv"] if spec.get("trades_csv") else None
    data = load_day(prices_path, trades_path)

    config = BacktestConfig(
        position_limits=spec["position_limits"],
        seed=spec.get("seed"),
    )
    result = run_backtest(trader, data, config)

    from backtester.metrics import compute_metrics

    metrics = compute_metrics(result)
    live_pnl = float(spec["final_pnl_live"])
    tol_pct = float(spec.get("tolerance_pnl_pct", DEFAULT_PNL_TOLERANCE_PCT))
    if live_pnl == 0:
        assert abs(metrics.final_pnl) <= max(1.0, tol_pct * 100), (
            f"Live PnL is 0; backtest PnL must be small: got {metrics.final_pnl}"
        )
    else:
        rel = abs(metrics.final_pnl - live_pnl) / abs(live_pnl)
        assert rel <= tol_pct, (
            f"PnL drift {rel:.4f} > tolerance {tol_pct} "
            f"(live={live_pnl}, backtest={metrics.final_pnl})"
        )

    # Position trajectory check.
    expected_traj = spec.get("position_trajectory") or []
    by_ts: Dict[int, Dict[str, int]] = {
        entry["timestamp"]: entry["positions"] for entry in expected_traj
    }
    for log in result.tick_logs:
        if log.timestamp not in by_ts:
            continue
        for product, expected_pos in by_ts[log.timestamp].items():
            actual = log.positions.get(product, 0)
            drift = abs(actual - expected_pos)
            assert drift <= POSITION_TOLERANCE, (
                f"Position drift at ts={log.timestamp} {product}: "
                f"actual={actual} expected={expected_pos} drift={drift} > {POSITION_TOLERANCE}"
            )


def test_replay_harness_is_importable() -> None:
    """Even with no fixtures present, the harness must import cleanly."""
    # If we reached this line, the module imported; that's the assertion.
    assert True
