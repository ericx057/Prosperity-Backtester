# Prosperity Backtester v1

Fidelity-faithful offline replay for IMC Prosperity. Built to support four
workflows:

1. Plateau parameter sweeps (anti-overfitting discipline).
2. Strategy variant comparison.
3. Pre-submission regression checks.
4. Offline detection of position-limit violations, trader exceptions, and
   timeouts.

This is not a GUI, not a dashboard, not a live-trading bridge. It is a
command-line backtester whose matching engine has been pinned against the
live simulator via a fidelity fixture suite.

---

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Or, for editable installs:

```bash
pip install -e .
```

## Quick start

Run a single-day backtest on the bundled synthetic fixture:

```bash
python backtest.py \
    --data backtester/tests/fixtures/prices_synthetic_day.csv \
    --trades backtester/tests/fixtures/trades_synthetic_day.csv \
    --trader example_trader.py \
    --out out/backtest
```

Outputs in `out/backtest/`:

- `results.json` - machine-readable per-tick fills, positions, warnings,
  rejections, and metrics.
- `summary.png` - 4-panel matplotlib summary (PnL curve, position, fill
  scatter, drawdown).

Run a parameter sweep:

```bash
python sweep.py \
    --data backtester/tests/fixtures/prices_synthetic_day.csv \
    --trades backtester/tests/fixtures/trades_synthetic_day.csv \
    --sweep-config backtester/tests/fixtures/sweep_config.yaml \
    --out out/sweep \
    --workers 4
```

Outputs in `out/sweep/`:

- `sweep_results.csv` - one row per parameter combination with final PnL,
  max drawdown, Sharpe, and all other metrics.
- `heatmap_<metric>_<dim1>_vs_<dim2>.png` - one heatmap per pair of
  swept dimensions.

## Project layout

```
backtester/
    datamodel.py          # official Prosperity types (Order, OrderDepth, TradingState, ...)
    data_loader.py        # CSV -> per-tick OrderDepth snapshots
    matching_engine.py    # pure, fidelity-critical (no deps on runner/metrics/reporter)
    runner.py             # tick loop, calls trader.run()
    metrics.py            # PnL, drawdown, Sharpe, fill quality
    reporter.py           # JSON + matplotlib 4-panel
    sweeper.py            # grid search w/ multiprocessing + heatmaps
    tests/
        fidelity/         # pins matching engine behavior against live
        unit/             # standard module tests
        fixtures/         # synthetic CSVs + sweep config
backtest.py               # CLI: single-day backtest
sweep.py                  # CLI: parameter sweep
example_trader.py         # deterministic market-maker used by smoke/fixture tests
pyproject.toml
requirements.txt
Makefile
.fidelity_lock            # sha256 of matching_engine.py when fidelity suite last passed
```

## Architecture invariants

- `matching_engine.py` has zero imports from `runner`, `metrics`,
  `reporter`, `sweeper`, or `data_loader`. Enforced by
  `tests/fidelity/test_module_boundary.py`.
- All datamodel types live in `backtester.datamodel` and match the
  upstream Prosperity spec verbatim. A trader that does
  `from datamodel import Order, OrderDepth, ...` works unchanged because
  `backtest.py` installs a `sys.modules["datamodel"]` shim.
- `matching_engine.py` is a pure function:
  `match(book, orders, position, limits) -> (trades, new_book, new_position, rejections)`.
  It never mutates the input `OrderDepth`.

## Fidelity doctrine

The matching engine pins these invariants (PRD section 4). Each one is
exercised by at least one fidelity fixture test.

1. **Within-tick sequence.** For each tick: the user's previous orders
   are implicitly cancelled (the runner rebuilds state every tick); the
   NPC book is populated from the CSV; user orders match against that
   resulting book. We never match against a pre-cancel book.
2. **Maker vs. taker.** A user order that crosses the existing book is
   a taker -- fills immediately at book prices. A user order that does
   not cross is a maker -- it sits in the spread and fills only if a
   market trade for the tick passes through its price.
3. **Fill price = resting order's price, NOT user's price.** Taking
   against a level at 10004 fills at 10004 even if the user bid 10010.
   The only exception is the market-trade pass-through path, where fill
   price follows the user's order (they get the improvement vs. the NPC
   counterparty).
4. **Aggregate position-limit check.** We sum longs and shorts before
   any execution; if any partial-fill scenario would breach, we reject
   the ENTIRE product batch. Per-order checking is a known naive bug
   and is explicitly tested against.
5. **traderData round-trip.** Every tick forces the returned traderData
   through `json.dumps`. Non-serializable payloads, or payloads over
   the 1 MiB cap, reset traderData to `""` and log a warning.
6. **Timeout / exception.** `trader.run()` over 900 ms or raising
   drops all orders for the tick and leaves position unchanged. `run()`
   over 500 ms is flagged as a yellow warning. Both are pinned by
   fidelity fixtures.
7. **Zero / invalid orders.** Zero-quantity orders are rejected.
   Non-integer price or quantity raises `ValueError` (matches the live
   `type_check_orders` behavior).
8. **Self-cross.** If a single tick's batch contains a user buy and a
   user sell whose prices cross, the entire batch is rejected. This is
   a deliberate PRD default -- the live exchange does not internally
   cross two user orders, and self-trading would poison PnL math.

### The fidelity lock

`.fidelity_lock` holds the sha256 of `backtester/matching_engine.py`
taken at the moment the fidelity suite last ran green. When the engine
changes:

```bash
make fidelity   # all fidelity tests must pass
python3 -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('backtester/matching_engine.py').read_bytes()).hexdigest())" > .fidelity_lock
```

If the lock hash does not match on startup, `backtest.py` and
`sweep.py` print a loud warning. Do NOT trust sweep results without a
fresh lock.

## Plateau rule (anti-overfitting)

Sweep results are intentionally NOT auto-filtered to a "best" config.
Plateau analysis is performed offline:

1. Load `sweep_results.csv` into a notebook.
2. For each candidate parameter tuple, identify a local neighborhood
   (+/-10% on each numeric dimension).
3. A plateau qualifies only if at least 80% of the neighborhood has a
   metric within 10% of the peak.
4. Select parameters from the interior of the widest plateau.

Isolated PnL spikes are rejected. If a strategy only works at one exact
parameter tuple, it is not an edge; it is overfit.

## Replay-equivalence harness

`backtester/tests/fidelity/test_replay_equivalence.py` asserts that an
offline backtest produces:

- Final PnL within +/-2% of the live Prosperity simulator.
- Position at every tick within +/-1 unit of the live trajectory.

The harness is implemented and runnable. It has no fixtures by default
because we have no live logs yet. To add one:

1. Run a deterministic trader on the live platform.
2. Capture the prices CSV, trades CSV, and per-tick activity log from
   the platform download.
3. Create `backtester/tests/fidelity/replay_fixtures/{name}/` with:
   - `prices_round_N_day_M.csv`
   - `trades_round_N_day_M.csv`
   - `trader.py` (or copy the trader into the fixture dir)
   - `expected.json` (schema in the module docstring)
4. `pytest -m replay` -- the case will auto-discover.

### TODO: populate replay fixtures

- [ ] Round 1 deterministic market-maker on RESIN
- [ ] Round 2 basket arbitrage trader
- [ ] Round 3 mean-reversion on KELP

Each new fixture must be tagged with the platform version it was
captured against. If the platform changes its matching rules, treat the
fixtures as versioned -- keep old ones and add new ones rather than
overwriting.

## Running tests

```bash
make test         # full suite
make unit         # unit-marked tests only
make fidelity     # fidelity-marked tests only (matching engine invariants)
pytest -m "not slow"   # skip long-running tests
pytest --update-snapshots backtester/tests/fidelity/test_regression_snapshots.py
```

When you intentionally change matching-engine behavior, run
`--update-snapshots` and review the diff before committing.

## Writing a trader

Your trader is a Python file at any path. It must define a class named
`Trader` with a `run(state: TradingState) -> (orders, conversions, traderData)`
method, matching the official Prosperity spec.

Minimum example:

```python
from datamodel import Order, TradingState

class Trader:
    def run(self, state: TradingState):
        orders = {}
        if "RESIN" in state.order_depths:
            orders["RESIN"] = [Order("RESIN", 9999, 5), Order("RESIN", 10001, -5)]
        return orders, 0, state.traderData
```

The `from datamodel import ...` line works because `backtest.py`
installs a `sys.modules["datamodel"]` shim pointing at
`backtester.datamodel`.

## Determinism

- Matching engine: purely deterministic.
- Runner: seeds both `random` and `numpy.random` when `seed` is set in
  the run config or sweep config.
- For byte-identical PnL across machines, run with
  `PYTHONHASHSEED=0 python backtest.py ...`. Dictionary ordering is
  stable in CPython 3.7+ but hash-randomization can still change string
  key ordering in traderData serialization.

## Known limitations (out of scope for v1)

- Conversions (Round 4+): stubbed. The runner emits `conversions=0` and
  does not process them.
- Options pricing / volatility smiles (Round 5): not implemented. The
  backtester treats option symbols as generic products.
- Multi-round chaining: one day per run.
- Tape deanonymization: separate tool.
- Distributed sweeps: single-machine multiprocessing only.
- GUI / web dashboard: none. CLI + matplotlib PNGs only.

## License attribution

This project was informed by the MIT-licensed
[jmerle/imc-prosperity-3-backtester](https://github.com/jmerle/imc-prosperity-3-backtester)
reference implementation, especially the datamodel types and the
structure of the match loop. All code in this repository was written
fresh; see `backtester/datamodel.py` for the verbatim datamodel port
(MIT upstream).
