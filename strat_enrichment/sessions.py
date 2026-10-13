"""Session anchoring: which trading session each 1-minute bar belongs to.

Every timeframe is anchored to the session open, not to midnight, so bars line
up with how the market trades:

- futures (CME): the session opens at 18:00 ET and is dated by the day it
  closes. Sunday 18:00 ET opens Monday's session.
- stocks: regular hours only, 09:30 to 16:00 ET. Pre- and post-market minutes
  are dropped.

Times are ET wall clock, so the anchor follows daylight saving time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"


@dataclass(frozen=True)
class Market:
    anchor: int                 # session open, minutes after midnight ET
    hours: int | None = None    # session length in minutes; None = trades around the clock
    overnight: bool = False     # session opens the evening before the day it is dated


MARKETS = {
    "futures": Market(anchor=18 * 60, overnight=True),
    "stocks": Market(anchor=9 * 60 + 30, hours=390),
}


def sessionize(minutes: pd.DataFrame, market: str) -> pd.DataFrame:
    """Add `session` (session date) and `since` (minutes since the session open).

    `minutes` needs a `ts` column: bar start, UTC epoch seconds. Minutes outside
    the market's session hours are dropped.
    """
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; use one of {sorted(MARKETS)}")
    m = MARKETS[market]
    et = pd.to_datetime(minutes["ts"].to_numpy(), unit="s", utc=True).tz_convert(ET)
    clock = (et.hour * 60 + et.minute).to_numpy()
    day = et.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")

    if m.overnight:
        session = day + (clock >= m.anchor).astype("timedelta64[D]")
        since = (clock - m.anchor) % 1440
    else:
        session = day
        since = clock - m.anchor

    out = minutes.assign(session=session, since=since)
    if m.hours is not None:
        out = out[(since >= 0) & (since < m.hours)]
    return out.reset_index(drop=True)
