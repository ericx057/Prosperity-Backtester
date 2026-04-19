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
    round2.py             # Round 2 MAF auction (Round2Config, resolve_maf_auction)
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

## Round 2: Market Access Fee (MAF)

Round 2 of IMC Prosperity 4 (2026) adds a pay-to-win auction on top of the
Round 1 trading loop:

> "You may include a Market Access Fee (MAF) in your Python program to gain
> access to an additional 25% of quotes. Only the top 50% of total MAFs will
> secure the contract, pay the MAF, and access this additional 25%. Others
> will not have to pay their MAF and will continue trading with the original
> volume allocation."

The backtester integrates this as an opt-in auction wrapper around the
matching engine. None of the Round 2 logic leaks into the Round 1 execution
path; a trader with no MAF declaration and no `--round2-config` produces
byte-identical output to the pre-Round-2 runner.

### Configurable field names and magic numbers

Per explicit project rule, the backtester does not hardcode any Round 2
variable name or numeric magic. Every knob lives on `Round2Config`
(`backtester/round2.py`) and is overridable via YAML:

| Round2Config field              | Default      | Purpose                                                        |
| ------------------------------- | ------------ | -------------------------------------------------------------- |
| `enabled`                       | `False`      | Master switch. Off = identical to Round 1.                     |
| `maf_method_name`               | `"get_maf"`  | Method on the Trader the runner calls to get the MAF bid.      |
| `maf_attribute_name`            | `"maf"`      | Fallback attribute if the method is absent.                    |
| `maf_field_name`                | `"MAF"`      | Fallback field inside (JSON-decoded) traderData.               |
| `volume_boost_pct`              | `0.25`       | Multiplier for winner's visible book depth.                    |
| `winner_top_fraction`           | `0.5`        | Top fraction of the MAF field that wins the auction.           |
| `auction_mode`                  | `"threshold"`| `"threshold"`, `"distribution"`, `"always_win"`, `"always_lose"`. |
| `competition_threshold`         | `0.0`        | (`threshold` mode) Trader wins iff MAF >= threshold.           |
| `competition_sample_size`       | `19`         | (`distribution` mode) Number of competing MAFs sampled.        |
| `competition_mean` / `_std`     | `0.0` / `1.0`| (`distribution` mode) `N(mean, std)` for competing bids.       |
| `auction_seed`                  | `None`       | RNG seed for distribution mode; falls back to backtest `seed`. |
| `auction_frequency`             | `"tick"`     | `"tick"` (re-auction each tick) or `"once"` (cache outcome).   |

### Declaring a MAF in your trader

Preferred: expose a `get_maf(state)` method. The runner calls it once per
tick and supports three return shapes:

```python
class Trader:
    def run(self, state):
        ...

    def get_maf(self, state) -> float:
        # Single scalar = round-level MAF, applied to every product.
        return 7.5

    # Or return a per-product mapping:
    def get_maf(self, state) -> dict:
        return {"RESIN": 10.0, "KELP": 2.0}
```

The method name is configurable. If your trader uses a different convention:

```yaml
# round2_config.yaml
maf_method_name: compute_maf
```

Fallback paths (tried in order if the method is absent):

1. `trader.<maf_attribute_name>` - a scalar or mapping attribute.
2. `traderData[<maf_field_name>]` - a field inside the JSON-encoded
   traderData string returned from `run()`.

If none of these resolve a value, the tick has no MAF declaration (no boost,
no fee).

### Running a Round 2 backtest

```bash
python backtest.py \
    --data backtester/tests/fixtures/prices_synthetic_day.csv \
    --trader example_trader.py \
    --round2-config backtester/tests/fixtures/round2_config.yaml \
    --out out/backtest_r2
```

The stdout summary gains a `round2:` line (`fees_paid=X.XX  wins=Y/Z`) and
`results.json` gains a `round2` block with per-tick auction outcomes and
fees.

### Sweeping Round 2 parameters

Parameter names prefixed with `round2.` are routed into the Round 2 config
for that combo. Everything else flows to the trader factory as before.

Example: `backtester/tests/fixtures/round2_sweep_config.yaml` sweeps both
trader knobs and auction knobs together. The sweep CSV adds
`total_fees_paid` and `net_pnl = final_pnl - total_fees_paid` columns so
plateau analysis can be done on net PnL directly:

```bash
python sweep.py \
    --data backtester/tests/fixtures/prices_synthetic_day.csv \
    --sweep-config backtester/tests/fixtures/round2_sweep_config.yaml \
    --out out/sweep_r2
```

### Auction simulation assumptions

The backtester has no live visibility into competitors' bids. The three
auction modes trade off different forms of uncertainty:

- `threshold`: Simplest. You win iff your MAF is at least
  `competition_threshold`. Good for sensitivity analysis ("would I be
  net-positive at any plausible clearing price?") but does not model
  uncertainty in that clearing price.
- `distribution`: Samples `competition_sample_size` competing MAFs from
  `N(mean, std)` and ranks your bid against the full field. Seeded for
  determinism. Models uncertainty but the shape of the competitor
  distribution is a free parameter.
- `always_win` / `always_lose`: Pinned outcomes, useful for tests and for
  isolating "do I benefit from +25% at all" from "will I clear the auction."

Use Round 2 backtests for sensitivity analysis, not absolute PnL prediction.

### Open questions (mechanic ambiguities)

The competition wiki was not publicly indexed during implementation. The
following were answered with defensible defaults, subject to confirmation
from official docs:

1. **Per-round vs. per-product MAF?** Default: per-product, via a dict return
   from `get_maf`. A scalar return is treated as a round-level MAF applied to
   every product. Covers both interpretations.
2. **What is "+25% of quotes"?** Default: NPC book depth boost (more
   counterparty liquidity to trade against). Alternative (bigger user
   orders) is not implemented - the matching engine's `volume_multiplier`
   scales the book only.
3. **Tie-breaking in "top 50%"?** Default: strict. 20 teams, trader ranked
   10th does not win; 9th does.
4. **MAF value type?** Default: float. An integer bid is the trivial
   sub-case.

Change any of these by editing `Round2Config` or adjusting `auction_mode` /
`auction_frequency`.

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
