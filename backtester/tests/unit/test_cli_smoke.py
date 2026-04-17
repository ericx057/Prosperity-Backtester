"""End-to-end smoke test: the ``backtest.py`` and ``sweep.py`` CLIs run to
completion on synthetic fixture data and produce their documented artifacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "backtester" / "tests" / "fixtures"


def test_backtest_cli_produces_outputs(tmp_path: Path) -> None:
    out = tmp_path / "bt"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "backtest.py"),
            "--data",
            str(FIXTURES / "prices_synthetic_day.csv"),
            "--trades",
            str(FIXTURES / "trades_synthetic_day.csv"),
            "--trader",
            str(ROOT / "example_trader.py"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "results.json").exists()
    assert (out / "summary.png").exists()


def test_sweep_cli_produces_outputs(tmp_path: Path) -> None:
    out = tmp_path / "sw"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sweep.py"),
            "--data",
            str(FIXTURES / "prices_synthetic_day.csv"),
            "--trades",
            str(FIXTURES / "trades_synthetic_day.csv"),
            "--sweep-config",
            str(FIXTURES / "sweep_config.yaml"),
            "--out",
            str(out),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "sweep_results.csv").exists()
    # At least one heatmap.
    heatmaps = list(out.glob("heatmap_*.png"))
    assert len(heatmaps) >= 1
