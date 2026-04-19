"""Unit tests for backtester.round2 - Round 2 MAF auction primitives.

Covers Round2Config validation, resolve_maf_auction across all auction
modes, scalar vs per-product MAFs, and YAML loading.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml

from backtester.round2 import (
    MAFAuctionResult,
    Round2Config,
    get_scalar_result,
    is_scalar_result,
    load_round2_config_from_yaml,
    resolve_maf_auction,
    round2_config_from_dict,
)

pytestmark = pytest.mark.unit


# ---------- Round2Config defaults & overrides ----------


class TestRound2ConfigDefaults:
    def test_defaults_match_round2_announcement(self) -> None:
        cfg = Round2Config()
        # +25% volume boost per announcement.
        assert cfg.volume_boost_pct == 0.25
        # Top 50% win.
        assert cfg.winner_top_fraction == 0.5
        # Off by default - Round 1 behavior must be preserved unless opted in.
        assert cfg.enabled is False

    def test_configurable_method_name(self) -> None:
        cfg = Round2Config(maf_method_name="compute_maf")
        assert cfg.maf_method_name == "compute_maf"

    def test_configurable_field_name(self) -> None:
        cfg = Round2Config(maf_field_name="bid_fee")
        assert cfg.maf_field_name == "bid_fee"

    def test_configurable_volume_boost(self) -> None:
        cfg = Round2Config(volume_boost_pct=0.50)
        assert cfg.volume_boost_pct == 0.50

    def test_configurable_winner_fraction(self) -> None:
        cfg = Round2Config(winner_top_fraction=0.3)
        assert cfg.winner_top_fraction == 0.3


class TestRound2ConfigValidation:
    def test_invalid_auction_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="auction_mode"):
            Round2Config(auction_mode="nonsense")

    def test_invalid_auction_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="auction_frequency"):
            Round2Config(auction_frequency="nonsense")

    def test_negative_boost_raises(self) -> None:
        with pytest.raises(ValueError, match="volume_boost_pct"):
            Round2Config(volume_boost_pct=-0.1)

    def test_winner_fraction_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="winner_top_fraction"):
            Round2Config(winner_top_fraction=0.0)

    def test_winner_fraction_one_raises(self) -> None:
        with pytest.raises(ValueError, match="winner_top_fraction"):
            Round2Config(winner_top_fraction=1.0)

    def test_negative_sample_size_raises(self) -> None:
        with pytest.raises(ValueError, match="competition_sample_size"):
            Round2Config(competition_sample_size=-1)

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="competition_std"):
            Round2Config(competition_std=-1.0)


# ---------- resolve_maf_auction ----------


class TestThresholdMode:
    def test_at_or_above_threshold_wins(self) -> None:
        cfg = Round2Config(
            enabled=True, auction_mode="threshold", competition_threshold=5.0
        )
        rng = random.Random(0)
        results = resolve_maf_auction(10.0, cfg, rng)
        result = get_scalar_result(results)
        assert result.won is True
        assert result.fee_paid == 10.0
        assert result.volume_multiplier == 1.25

    def test_below_threshold_loses(self) -> None:
        cfg = Round2Config(
            enabled=True, auction_mode="threshold", competition_threshold=5.0
        )
        rng = random.Random(0)
        results = resolve_maf_auction(3.0, cfg, rng)
        result = get_scalar_result(results)
        assert result.won is False
        assert result.fee_paid == 0.0
        assert result.volume_multiplier == 1.0

    def test_zero_maf_never_wins(self) -> None:
        cfg = Round2Config(
            enabled=True, auction_mode="threshold", competition_threshold=0.0
        )
        rng = random.Random(0)
        results = resolve_maf_auction(0.0, cfg, rng)
        result = get_scalar_result(results)
        assert result.won is False
        assert result.fee_paid == 0.0
        assert result.volume_multiplier == 1.0

    def test_deterministic_same_bid_same_outcome(self) -> None:
        """Threshold mode is pure; same input always gives same result."""
        cfg = Round2Config(
            enabled=True, auction_mode="threshold", competition_threshold=5.0
        )
        for _ in range(5):
            rng = random.Random(42)
            result = get_scalar_result(resolve_maf_auction(7.0, cfg, rng))
            assert result.won is True


class TestDistributionMode:
    def test_same_seed_same_outcome(self) -> None:
        cfg = Round2Config(
            enabled=True,
            auction_mode="distribution",
            competition_sample_size=19,
            competition_mean=5.0,
            competition_std=2.0,
        )
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        r1 = get_scalar_result(resolve_maf_auction(6.0, cfg, rng1))
        r2 = get_scalar_result(resolve_maf_auction(6.0, cfg, rng2))
        assert r1.won == r2.won
        assert r1.fee_paid == r2.fee_paid

    def test_high_bid_almost_always_wins(self) -> None:
        """With competitors N(0, 1), a bid of 100 should always win."""
        cfg = Round2Config(
            enabled=True,
            auction_mode="distribution",
            competition_sample_size=19,
            competition_mean=0.0,
            competition_std=1.0,
        )
        wins = 0
        for seed in range(50):
            rng = random.Random(seed)
            if get_scalar_result(resolve_maf_auction(100.0, cfg, rng)).won:
                wins += 1
        assert wins == 50

    def test_low_bid_almost_never_wins(self) -> None:
        """A bid far below the competition distribution mean should never win.

        Note: the minimum positive bid we can test with is a tiny epsilon,
        since 0 is pinned to lose via the zero-MAF guard.
        """
        cfg = Round2Config(
            enabled=True,
            auction_mode="distribution",
            competition_sample_size=19,
            competition_mean=10.0,
            competition_std=1.0,
        )
        wins = 0
        for seed in range(50):
            rng = random.Random(seed)
            if get_scalar_result(resolve_maf_auction(0.01, cfg, rng)).won:
                wins += 1
        assert wins == 0


class TestTop50PercentRule:
    def test_rank_10_of_20_does_not_win(self) -> None:
        """20 teams total, trader's MAF ranks 10th (below top half) -> lose."""
        # We construct a scenario where exactly 10 of 19 competitors beat the
        # trader; this makes the trader's rank 10 / 20 (strictly inside bottom half).
        cfg = Round2Config(
            enabled=True,
            auction_mode="distribution",
            competition_sample_size=19,
        )
        # Patch random to emit deterministic MAFs.
        class PinnedRng:
            def __init__(self, values: list[float]) -> None:
                self.values = list(values)

            def gauss(self, mu: float, sigma: float) -> float:
                return self.values.pop(0)

        # Trader bids 5.0. Give 10 competitors above and 9 below.
        competitors = [6.0] * 10 + [4.0] * 9
        rng = PinnedRng(competitors)
        result = get_scalar_result(resolve_maf_auction(5.0, cfg, rng))
        # Trader's rank from top = 20 - 1 - strictly_below
        #   strictly_below = 9 (the 9 who bid 4.0)
        #   rank_from_top = 10 (0-indexed)
        # winners = floor(0.5 * 20) = 10
        # rank 10 < winners 10 -> False -> LOSE
        assert result.won is False

    def test_rank_9_of_20_wins(self) -> None:
        cfg = Round2Config(
            enabled=True,
            auction_mode="distribution",
            competition_sample_size=19,
        )

        class PinnedRng:
            def __init__(self, values: list[float]) -> None:
                self.values = list(values)

            def gauss(self, mu: float, sigma: float) -> float:
                return self.values.pop(0)

        # Trader bids 5.0. 9 competitors above, 10 below -> trader rank = 9/20.
        competitors = [6.0] * 9 + [4.0] * 10
        rng = PinnedRng(competitors)
        result = get_scalar_result(resolve_maf_auction(5.0, cfg, rng))
        # strictly_below = 10, rank_from_top = 19 - 10 = 9
        # winners = 10, 9 < 10 -> WIN
        assert result.won is True


class TestAlwaysWinLose:
    def test_always_win_mode_returns_won_even_for_zero_bid(self) -> None:
        cfg = Round2Config(enabled=True, auction_mode="always_win")
        rng = random.Random(0)
        result = get_scalar_result(resolve_maf_auction(0.0, cfg, rng))
        assert result.won is True
        assert result.volume_multiplier == 1.25

    def test_always_lose_mode(self) -> None:
        cfg = Round2Config(enabled=True, auction_mode="always_lose")
        rng = random.Random(0)
        result = get_scalar_result(resolve_maf_auction(1_000_000.0, cfg, rng))
        assert result.won is False
        assert result.fee_paid == 0.0


class TestPerProductAuction:
    def test_dict_input_resolved_independently(self) -> None:
        cfg = Round2Config(
            enabled=True, auction_mode="threshold", competition_threshold=5.0
        )
        rng = random.Random(0)
        bids = {"RESIN": 10.0, "KELP": 2.0}
        results = resolve_maf_auction(bids, cfg, rng)
        assert not is_scalar_result(results)
        assert results["RESIN"].won is True
        assert results["KELP"].won is False

    def test_scalar_input_is_marked(self) -> None:
        cfg = Round2Config(enabled=True, auction_mode="threshold")
        rng = random.Random(0)
        results = resolve_maf_auction(10.0, cfg, rng)
        assert is_scalar_result(results)

    def test_none_input_yields_empty(self) -> None:
        cfg = Round2Config(enabled=True)
        rng = random.Random(0)
        results = resolve_maf_auction(None, cfg, rng)
        assert results == {}


# ---------- configurable boost + winner fraction ----------


class TestConfigurableBoost:
    def test_boost_50_percent(self) -> None:
        cfg = Round2Config(
            enabled=True,
            auction_mode="always_win",
            volume_boost_pct=0.50,
        )
        rng = random.Random(0)
        result = get_scalar_result(resolve_maf_auction(1.0, cfg, rng))
        assert result.volume_multiplier == 1.5

    def test_boost_zero(self) -> None:
        """Edge case: boost = 0 means winner pays fee but gets no boost."""
        cfg = Round2Config(
            enabled=True,
            auction_mode="always_win",
            volume_boost_pct=0.0,
        )
        rng = random.Random(0)
        result = get_scalar_result(resolve_maf_auction(1.0, cfg, rng))
        assert result.volume_multiplier == 1.0
        assert result.won is True


# ---------- YAML loading ----------


class TestYAMLLoading:
    def test_flat_yaml_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "r2.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "volume_boost_pct": 0.5,
                    "winner_top_fraction": 0.3,
                    "auction_mode": "threshold",
                    "competition_threshold": 7.5,
                }
            )
        )
        cfg = load_round2_config_from_yaml(path)
        assert cfg.enabled is True
        assert cfg.volume_boost_pct == 0.5
        assert cfg.winner_top_fraction == 0.3
        assert cfg.competition_threshold == 7.5

    def test_nested_yaml_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "r2.yaml"
        path.write_text(
            yaml.safe_dump(
                {"round2": {"enabled": True, "volume_boost_pct": 0.10}}
            )
        )
        cfg = load_round2_config_from_yaml(path)
        assert cfg.enabled is True
        assert cfg.volume_boost_pct == 0.10

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "r2.yaml"
        path.write_text(yaml.safe_dump({"unknown_key": "oops"}))
        with pytest.raises(TypeError):
            load_round2_config_from_yaml(path)

    def test_from_dict_strict(self) -> None:
        with pytest.raises(TypeError):
            round2_config_from_dict({"not_a_real_field": 1})
