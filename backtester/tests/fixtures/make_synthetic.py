"""Generate deterministic synthetic data for smoke tests.

Produces prices_synthetic_day.csv and trades_synthetic_day.csv in the same
directory as this script. The DGP: RESIN mid = 10000 + small bounded noise,
KELP random walk around 2000. The noise is deterministic (seed=0).

Run: python backtester/tests/fixtures/make_synthetic.py
"""

from __future__ import annotations

import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    random.seed(0)
    prices_path = HERE / "prices_synthetic_day.csv"
    trades_path = HERE / "trades_synthetic_day.csv"

    lines = [
        "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    ]
    trades_lines = ["timestamp;buyer;seller;symbol;currency;price;quantity"]

    kelp_mid = 2000
    for tick_idx in range(200):
        ts = tick_idx * 100
        # RESIN: mid jitters in [-2, 2] around 10000; bid 10000-1, ask 10000+1.
        resin_noise = random.randint(-2, 2)
        resin_mid = 10000 + resin_noise * 0.0  # stays at 10000
        resin_bid_px = 9998
        resin_ask_px = 10002
        resin_bid_vol = random.randint(3, 8)
        resin_ask_vol = random.randint(3, 8)
        lines.append(
            ";".join(
                [
                    "0",
                    str(ts),
                    "RESIN",
                    str(resin_bid_px),
                    str(resin_bid_vol),
                    "",
                    "",
                    "",
                    "",
                    str(resin_ask_px),
                    str(resin_ask_vol),
                    "",
                    "",
                    "",
                    "",
                    f"{resin_mid + 10000:.1f}" if False else "10000.0",
                    "0",
                ]
            )
        )

        # KELP: random walk +/- 1 per tick.
        step = random.choice([-1, 0, 0, 1])
        kelp_mid += step
        kelp_bid_px = kelp_mid - 1
        kelp_ask_px = kelp_mid + 1
        kelp_bid_vol = random.randint(2, 6)
        kelp_ask_vol = random.randint(2, 6)
        lines.append(
            ";".join(
                [
                    "0",
                    str(ts),
                    "KELP",
                    str(kelp_bid_px),
                    str(kelp_bid_vol),
                    "",
                    "",
                    "",
                    "",
                    str(kelp_ask_px),
                    str(kelp_ask_vol),
                    "",
                    "",
                    "",
                    "",
                    f"{kelp_mid}.0",
                    "0",
                ]
            )
        )

        # Occasional NPC trades.
        if tick_idx % 7 == 0:
            trades_lines.append(
                f"{ts};NPC_A;NPC_B;RESIN;SHELLS;{resin_ask_px};2"
            )
        if tick_idx % 11 == 0:
            trades_lines.append(
                f"{ts};NPC_C;NPC_D;KELP;SHELLS;{kelp_bid_px};1"
            )

    prices_path.write_text("\n".join(lines) + "\n")
    trades_path.write_text("\n".join(trades_lines) + "\n")

    print(f"wrote {prices_path}")
    print(f"wrote {trades_path}")


if __name__ == "__main__":
    main()
