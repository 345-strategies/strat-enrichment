"""Synthetic 1-minute bars for trying the pipeline. Random walk, not market data.

    python -m strat_enrichment.sample sample/futures_1m.parquet --market futures

Futures: CME-style hours (Sunday 18:00 to Friday 17:00 ET, daily halt 17:00 to
18:00), 0.25 ticks, and a roll from SYNH6 to SYNM6 on 2026-03-18 00:00 ET where
the new contract trades 42 points higher, so back-adjustment has a gap to
remove. Stocks: 09:30 to 16:00 ET, one-cent ticks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .sessions import ET

ROLL_ET = "2026-03-18 00:00"
CARRY = 42.0


def synthetic_minutes(
    market: str = "futures",
    start: str = "2025-11-30",
    end: str = "2026-04-04",
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wall = pd.date_range(start, end, freq="1min", inclusive="left")
    clock = wall.hour * 60 + wall.minute
    dow = wall.dayofweek  # Monday 0 .. Sunday 6
    if market == "futures":
        day_part = (dow <= 4) & (clock < 17 * 60)             # Mon-Fri until 17:00
        evening = ((dow <= 3) | (dow == 6)) & (clock >= 18 * 60)  # Sun-Thu from 18:00
        keep = day_part | evening
        tick, price, opening = 0.25, 5000.0, clock == 18 * 60
    elif market == "stocks":
        keep = (dow <= 4) & (clock >= 570) & (clock < 960)
        tick, price, opening = 0.01, 500.0, clock == 570
    else:
        raise ValueError(f"unknown market {market!r}")
    wall, clock, opening = wall[keep], clock[keep], opening[keep]

    n = len(wall)
    busy = (clock >= 570) & (clock < 960)
    scale = price / 5000 * 0.25 / tick                        # same volatility in percent
    sigma = np.where(busy, 3.0, 1.2) * scale                  # ticks per minute
    drift = np.repeat(rng.normal(0, 0.15, n // 390 + 1), 390)[:n]
    step = np.round(rng.normal(drift, sigma)).astype(np.int64)
    gap = np.where(opening, np.round(rng.normal(0, 8 * scale, n)), 0).astype(np.int64)
    close = int(round(price / tick)) + np.cumsum(step + gap)
    open_ = np.r_[close[0], close[:-1]] + gap
    up = rng.poisson(0.6 * sigma)
    down = rng.poisson(0.6 * sigma)
    high = np.maximum(open_, close) + up
    low = np.minimum(open_, close) - down

    out = pd.DataFrame(
        {
            "ts": wall.tz_localize(ET).as_unit("s").asi8,
            "o": open_ * tick,
            "h": high * tick,
            "l": low * tick,
            "c": close * tick,
        }
    )
    if market == "futures":
        rolled = wall >= pd.Timestamp(ROLL_ET)
        out["contract"] = np.where(rolled, "SYNM6", "SYNH6")
        out.loc[rolled, ["o", "h", "l", "c"]] += CARRY
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--market", choices=("futures", "stocks"), default="futures")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    df = synthetic_minutes(a.market, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {len(df):,} synthetic minutes to {a.out}")


if __name__ == "__main__":
    main()
