"""Round 2: Market Access Fee (MAF) auction support.

Round 2 of IMC Prosperity 4 (2026) adds an auction:

    "You may include a Market Access Fee (MAF) in your Python program to gain
     access to an additional 25% of quotes. Only the top 50% of total MAFs
     will secure the contract, pay the MAF, and access this additional 25%.
     Others will not have to pay their MAF and will continue trading with
     the original volume allocation."

Design rules:

- No magic strings. The method name the trader exposes (``get_maf`` by default),
  the attribute/field name used as a fallback (``MAF`` by default), and every
  numeric threshold (``+25%``, ``top 50%``, etc.) are all overridable via
  ``Round2Config``.
- No magic numbers in the matching engine. The engine receives a
  ``volume_multiplier`` at each ``match()`` call. It never hardcodes ``1.25``.
- Backward compatible. A ``BacktestConfig`` with ``round2=None`` produces
  byte-identical output to the pre-Round-2 runner.

Offline auction simulation:

The backtester only has access to one trader. Competing MAFs are not
observable offline. ``Round2Config.auction_mode`` chooses between:

- ``"threshold"``: the trader wins the auction iff their MAF is at least
  ``competition_threshold``. The simplest baseline.
- ``"distribution"``: sample ``competition_sample_size`` competing MAFs from
  ``N(competition_mean, competition_std)``; rank the trader's MAF against the
  full field; win iff rank is strictly inside the top
  ``winner_top_fraction``.
- ``"always_win"`` / ``"always_lose"``: pinned outcomes. Useful for tests.

Public types:

- ``Round2Config`` - frozen dataclass of all Round 2 knobs.
- ``MAFAuctionResult`` - one auction outcome: ``won``, ``fee_paid``,
  ``volume_multiplier``.
- ``resolve_maf_auction`` - pure function, deterministic for a given seed.
- ``load_round2_config_from_yaml`` / ``round2_config_from_dict`` - YAML loader.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


# Sentinel product key used when the trader declares a single scalar MAF
# instead of a per-product mapping. Not configurable: it is an internal
# implementation detail not exposed to the trader.
_SCALAR_PRODUCT_KEY = "__round_level__"


@dataclass(frozen=True)
class Round2Config:
    """Configuration for the Round 2 MAF auction.

    Every name and every number here is overridable. Defaults mirror the
    official Round 2 announcement (+25% volume, top 50% win).
    """

    enabled: bool = False

    # --- configurable NAMES (no hardcoded strings in the runner) ---
    maf_method_name: str = "get_maf"
    maf_field_name: str = "MAF"
    maf_attribute_name: str = "maf"

    # --- configurable NUMBERS (no hardcoded magic in the engine) ---
    volume_boost_pct: float = 0.25
    winner_top_fraction: float = 0.5

    # --- auction simulation ---
    auction_mode: str = "threshold"
    competition_threshold: float = 0.0
    competition_sample_size: int = 19
    competition_mean: float = 0.0
    competition_std: float = 1.0
    auction_seed: Optional[int] = None

    # --- per-tick vs per-round auction ---
    # "tick": re-run the auction every tick (MAF value can float).
    # "once": run the auction once using the first-seen MAF.
    auction_frequency: str = "tick"

    def __post_init__(self) -> None:
        if self.auction_mode not in {
            "threshold",
            "distribution",
            "always_win",
            "always_lose",
        }:
            raise ValueError(
                f"auction_mode must be one of "
                f"'threshold', 'distribution', 'always_win', 'always_lose'; "
                f"got {self.auction_mode!r}"
            )
        if self.auction_frequency not in {"tick", "once"}:
            raise ValueError(
                f"auction_frequency must be 'tick' or 'once'; "
                f"got {self.auction_frequency!r}"
            )
        if self.volume_boost_pct < 0:
            raise ValueError(
                f"volume_boost_pct must be non-negative; "
                f"got {self.volume_boost_pct}"
            )
        if not 0.0 < self.winner_top_fraction < 1.0:
            raise ValueError(
                f"winner_top_fraction must be in (0, 1); "
                f"got {self.winner_top_fraction}"
            )
        if self.competition_sample_size < 0:
            raise ValueError(
                f"competition_sample_size must be non-negative; "
                f"got {self.competition_sample_size}"
            )
        if self.competition_std < 0:
            raise ValueError(
                f"competition_std must be non-negative; "
                f"got {self.competition_std}"
            )


@dataclass(frozen=True)
class MAFAuctionResult:
    """One product's auction outcome for one auction event."""

    won: bool
    fee_paid: float
    volume_multiplier: float
    declared_maf: float


def _rank_in_top(trader_maf: float, rng: random.Random, cfg: Round2Config) -> bool:
    """Sample competing MAFs and decide if trader is strictly in top fraction.

    Rule: trader wins iff fraction of competitors whose MAF is strictly less
    than the trader's MAF is >= (1 - winner_top_fraction). Equivalently, the
    trader is in the top ``winner_top_fraction`` of the combined field.

    Ties resolved by counting only strictly-smaller competitors (trader loses
    ties, consistent with "strict" top 50% interpretation).
    """
    n = cfg.competition_sample_size
    if n == 0:
        # No competitors - trader is the only participant; they win if MAF >= 0.
        return trader_maf >= 0.0
    competitors = [
        rng.gauss(cfg.competition_mean, cfg.competition_std) for _ in range(n)
    ]
    strictly_below = sum(1 for m in competitors if m < trader_maf)
    total_field = n + 1  # trader + competitors
    # trader_rank_from_top = total_field - 1 - strictly_below  (0-indexed from top)
    # winners = floor(winner_top_fraction * total_field)
    # trader wins iff trader_rank_from_top < winners
    winners = int(cfg.winner_top_fraction * total_field)
    trader_rank_from_top = total_field - 1 - strictly_below
    return trader_rank_from_top < winners


def _resolve_single(
    trader_maf: float,
    cfg: Round2Config,
    rng: random.Random,
) -> MAFAuctionResult:
    """Resolve the auction for a single (product or round-level) bid."""
    if trader_maf <= 0.0 and cfg.auction_mode != "always_win":
        # A zero or negative bid cannot win (it's equivalent to not bidding).
        # always_win mode forces a win for pinned-outcome testing.
        return MAFAuctionResult(
            won=False,
            fee_paid=0.0,
            volume_multiplier=1.0,
            declared_maf=trader_maf,
        )

    won: bool
    if cfg.auction_mode == "threshold":
        won = trader_maf >= cfg.competition_threshold and trader_maf > 0.0
    elif cfg.auction_mode == "distribution":
        won = _rank_in_top(trader_maf, rng, cfg)
    elif cfg.auction_mode == "always_win":
        won = True
    elif cfg.auction_mode == "always_lose":
        won = False
    else:
        # __post_init__ guards this, but be explicit.
        raise ValueError(f"unreachable auction_mode: {cfg.auction_mode!r}")

    if won:
        return MAFAuctionResult(
            won=True,
            fee_paid=float(trader_maf),
            volume_multiplier=1.0 + cfg.volume_boost_pct,
            declared_maf=trader_maf,
        )
    return MAFAuctionResult(
        won=False,
        fee_paid=0.0,
        volume_multiplier=1.0,
        declared_maf=trader_maf,
    )


def resolve_maf_auction(
    trader_maf: Union[float, Mapping[str, float], None],
    cfg: Round2Config,
    rng: random.Random,
) -> Dict[str, MAFAuctionResult]:
    """Resolve the auction for this tick.

    Args:
        trader_maf: either a scalar (round-level MAF) or a mapping
            ``{symbol: maf}`` for a per-product MAF. ``None`` means the
            trader declared no MAF this tick.
        cfg: Round2Config.
        rng: deterministic random source (seed it upstream for reproducibility).

    Returns:
        dict keyed by product symbol, or by the internal scalar-product key if
        ``trader_maf`` was a scalar. Each value is a ``MAFAuctionResult``.
    """
    if trader_maf is None:
        return {}

    if isinstance(trader_maf, Mapping):
        out: Dict[str, MAFAuctionResult] = {}
        # Deterministic iteration order across runs.
        for symbol in sorted(trader_maf.keys()):
            maf_value = float(trader_maf[symbol])
            out[symbol] = _resolve_single(maf_value, cfg, rng)
        return out

    # Scalar input.
    maf_value = float(trader_maf)
    return {_SCALAR_PRODUCT_KEY: _resolve_single(maf_value, cfg, rng)}


def is_scalar_result(results: Mapping[str, MAFAuctionResult]) -> bool:
    """True if the result dict was produced from a scalar trader MAF."""
    return len(results) == 1 and _SCALAR_PRODUCT_KEY in results


def get_scalar_result(results: Mapping[str, MAFAuctionResult]) -> MAFAuctionResult:
    """Extract the single result from a scalar auction."""
    return results[_SCALAR_PRODUCT_KEY]


def round2_config_from_dict(raw: Mapping[str, Any]) -> Round2Config:
    """Build a Round2Config from a plain dict (YAML-friendly).

    Extra/unknown keys raise a ``TypeError`` so typos surface loudly.
    """
    # Use dataclass default for any key that's missing - simple passthrough.
    # Any unrecognized key triggers a TypeError from Round2Config.
    return Round2Config(**dict(raw))


def load_round2_config_from_yaml(path: Union[str, Path]) -> Round2Config:
    """Load a Round2Config from a YAML file.

    Supports two YAML shapes:
        1. Top-level Round2Config fields.
        2. A top-level ``round2:`` key holding those fields.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Round 2 config YAML at {path} must be a mapping at the top level."
        )
    if "round2" in raw:
        inner = raw["round2"]
        if not isinstance(inner, Mapping):
            raise ValueError(
                f"round2 block in {path} must be a mapping; got {type(inner)!r}"
            )
        return round2_config_from_dict(inner)
    return round2_config_from_dict(raw)
