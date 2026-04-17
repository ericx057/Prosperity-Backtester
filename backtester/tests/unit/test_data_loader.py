"""Tests for backtester.data_loader: CSV -> per-tick OrderDepth snapshots."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from backtester.data_loader import PriceRow, TickSnapshot, load_day

pytestmark = pytest.mark.unit


def _write_prices_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "prices.csv"
    p.write_text(dedent(content).lstrip("\n"))
    return p


def _write_trades_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "trades.csv"
    p.write_text(dedent(content).lstrip("\n"))
    return p


class TestLoadDayPricesOnly:
    def test_single_row_single_product(self, tmp_path: Path) -> None:
        content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;0;KELP;2000;5;1999;3;;;2001;4;2002;2;;;2000.5;0
        """
        prices_path = _write_prices_csv(tmp_path, content)
        data = load_day(prices_path)
        assert data.products == ["KELP"]
        assert 0 in data.snapshots
        snap = data.snapshots[0]
        assert isinstance(snap, TickSnapshot)
        assert "KELP" in snap.prices
        row = snap.prices["KELP"]
        assert row.bid_prices == (2000, 1999)
        assert row.bid_volumes == (5, 3)
        assert row.ask_prices == (2001, 2002)
        assert row.ask_volumes == (4, 2)
        assert row.mid_price == 2000.5

    def test_multiple_products_multiple_ticks(self, tmp_path: Path) -> None:
        content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;0;KELP;2000;5;;;;;2001;4;;;;;2000.5;0
            0;0;RESIN;10000;10;;;;;10001;10;;;;;10000.5;0
            0;100;KELP;2000;7;;;;;2001;6;;;;;2000.5;0
            0;100;RESIN;9999;10;;;;;10001;10;;;;;10000.0;0
        """
        prices_path = _write_prices_csv(tmp_path, content)
        data = load_day(prices_path)
        assert set(data.products) == {"KELP", "RESIN"}
        assert sorted(data.snapshots) == [0, 100]
        assert set(data.snapshots[0].prices) == {"KELP", "RESIN"}
        assert data.snapshots[100].prices["KELP"].bid_volumes == (7,)


class TestLoadDayWithTrades:
    def test_market_trades_attached_to_snapshot(self, tmp_path: Path) -> None:
        prices_content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;0;KELP;2000;5;;;;;2001;4;;;;;2000.5;0
            0;100;KELP;2000;5;;;;;2001;4;;;;;2000.5;0
        """
        trades_content = """
            timestamp;buyer;seller;symbol;currency;price;quantity
            0;NPC_A;NPC_B;KELP;SHELLS;2001;3
            100;NPC_C;NPC_D;KELP;SHELLS;2000;2
        """
        prices_path = _write_prices_csv(tmp_path, prices_content)
        trades_path = _write_trades_csv(tmp_path, trades_content)
        data = load_day(prices_path, trades_path)
        t0 = data.snapshots[0].market_trades["KELP"]
        assert len(t0) == 1
        assert t0[0].price == 2001
        assert t0[0].quantity == 3
        assert t0[0].buyer == "NPC_A"
        assert t0[0].seller == "NPC_B"
        assert t0[0].timestamp == 0
        t1 = data.snapshots[100].market_trades["KELP"]
        assert t1[0].price == 2000

    def test_missing_trades_csv_ok(self, tmp_path: Path) -> None:
        prices_content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;0;KELP;2000;5;;;;;2001;4;;;;;2000.5;0
        """
        prices_path = _write_prices_csv(tmp_path, prices_content)
        data = load_day(prices_path)
        assert data.snapshots[0].market_trades == {}


class TestBuildOrderDepth:
    def test_order_depth_sell_volumes_are_negative(self, tmp_path: Path) -> None:
        """OrderDepth uses NEGATIVE quantities for sell_orders per official spec."""
        content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;0;K;100;5;99;3;;;101;4;102;2;;;100.5;0
        """
        prices_path = _write_prices_csv(tmp_path, content)
        data = load_day(prices_path)
        od = data.snapshots[0].build_order_depth("K")
        assert od.buy_orders == {100: 5, 99: 3}
        assert od.sell_orders == {101: -4, 102: -2}

    def test_missing_product_returns_empty(self, tmp_path: Path) -> None:
        content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;0;K;100;5;;;;;101;4;;;;;100.5;0
        """
        prices_path = _write_prices_csv(tmp_path, content)
        data = load_day(prices_path)
        od = data.snapshots[0].build_order_depth("MISSING")
        assert od.buy_orders == {}
        assert od.sell_orders == {}


class TestPriceRow:
    def test_is_immutable(self) -> None:
        row = PriceRow(
            day=0,
            timestamp=0,
            product="K",
            bid_prices=(100,),
            bid_volumes=(5,),
            ask_prices=(101,),
            ask_volumes=(4,),
            mid_price=100.5,
            profit_loss=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            row.mid_price = 200.0  # type: ignore[misc]


class TestSortedTimestamps:
    def test_returns_timestamps_sorted(self, tmp_path: Path) -> None:
        content = """
            day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
            0;200;K;100;5;;;;;101;4;;;;;100.5;0
            0;0;K;100;5;;;;;101;4;;;;;100.5;0
            0;100;K;100;5;;;;;101;4;;;;;100.5;0
        """
        prices_path = _write_prices_csv(tmp_path, content)
        data = load_day(prices_path)
        assert data.timestamps() == [0, 100, 200]
