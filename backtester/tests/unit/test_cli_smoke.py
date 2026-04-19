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


def test_backtest_cli_with_round2_config(tmp_path: Path) -> None:
    """End-to-end: backtest CLI runs with --round2-config and emits round2 data."""
    out = tmp_path / "r2bt"
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
            "--round2-config",
            str(FIXTURES / "round2_config.yaml"),
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
    # stdout should mention the Round 2 summary line.
    assert "round2:" in result.stdout
    # results.json should contain a round2 block with the expected keys.
    import json as _json
    doc = _json.loads((out / "results.json").read_text())
    assert "round2" in doc
    assert "total_fees_paid" in doc["round2"]
    assert "auction_outcomes" in doc["round2"]


def test_sweep_cli_with_round2_dims(tmp_path: Path) -> None:
    """End-to-end: sweep CLI routes ``round2.*`` params into the Round 2 config.

    The sample sweep uses competition_threshold and volume_boost_pct as swept
    dimensions. The output CSV must expose net_pnl and total_fees_paid.
    """
    out = tmp_path / "r2sw"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sweep.py"),
            "--data",
            str(FIXTURES / "prices_synthetic_day.csv"),
            "--trades",
            str(FIXTURES / "trades_synthetic_day.csv"),
            "--sweep-config",
            str(FIXTURES / "round2_sweep_config.yaml"),
            "--out",
            str(out),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    csv_path = out / "sweep_results.csv"
    assert csv_path.exists()
    header = csv_path.read_text().splitlines()[0].split(",")
    assert "total_fees_paid" in header
    assert "net_pnl" in header
    assert "round2.competition_threshold" in header
    assert "round2.volume_boost_pct" in header
    # At least one heatmap written.
    heatmaps = list(out.glob("heatmap_*.png"))
    assert len(heatmaps) >= 1
