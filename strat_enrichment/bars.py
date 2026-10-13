"""Aggregate session-anchored 1-minute bars into higher-timeframe bars.

Intraday bars count from the session open: on futures a 4h bar starts at 18:00,
22:00, 02:00, 06:00, 10:00 and 14:00 ET; on stocks a 1h bar starts at 09:30,
10:30 and so on. D is the session, W the ISO week of the session date, M its
calendar month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAMES = ("15m", "30m", "1h", "4h", "D", "W", "M")


def tf_minutes(tf: str) -> int | None:
    """Length of an intraday timeframe in minutes, or None for D, W and M."""
    if tf in ("D", "W", "M"):
        return None
    unit = {"m": 1, "h": 60}.get(tf[-1:])
    if unit is None or not tf[:-1].isdigit() or int(tf[:-1]) == 0:
        raise ValueError(f"unknown timeframe {tf!r}: use Nm, Nh, D, W or M")
    return int(tf[:-1]) * unit


def bar_keys(minutes: pd.DataFrame, tf: str) -> np.ndarray:
    """One integer per minute that is equal for every minute of the same bar."""
    session = minutes["session"].to_numpy().astype("datetime64[D]")
    n = tf_minutes(tf)
    if n is not None:
        return session.astype(np.int64) * 10_000 + minutes["since"].to_numpy() // n
    if tf == "D":
        return session.astype(np.int64)
    if tf == "W":
        iso = pd.DatetimeIndex(session).isocalendar()
        return (iso["year"] * 100 + iso["week"]).to_numpy(dtype=np.int64)
    months = session.astype("datetime64[M]")
    return months.astype(np.int64)


def build_bars(minutes: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (bars, bar index of every minute).

    `minutes` must be sessionized and sorted by `ts`. Each bar records the
    positions of its first and last minute (`i0`, `i1`) so callers can walk the
    1-minute path inside it.
    """
    keys = bar_keys(minutes, tf)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:] - 1, len(keys) - 1]
    bar_index = np.repeat(np.arange(len(starts)), ends - starts + 1)
    h = minutes["h"].to_numpy(dtype=float)
    l = minutes["l"].to_numpy(dtype=float)
    ts = minutes["ts"].to_numpy()

    # Slots are ET wall clock. A session that trades through the autumn DST
    # change repeats an hour, which would merge two real hours into one bar.
    n = tf_minutes(tf)
    if n is not None:
        too_long = np.flatnonzero(ts - ts[starts][bar_index] >= n * 60)
        if len(too_long):
            at = pd.Timestamp(int(ts[too_long[0]]), unit="s", tz="UTC")
            raise ValueError(
                f"{tf}: a bar spans more than {tf} of real time at {at}. The data "
                "trades through a daylight saving change, which session-anchored bars cannot represent."
            )

    bars = pd.DataFrame(
        {
            "ts": ts[starts],
            "ts_last": ts[ends],
            "session": minutes["session"].to_numpy()[starts],
            "o": minutes["o"].to_numpy(dtype=float)[starts],
            "h": np.maximum.reduceat(h, starts),
            "l": np.minimum.reduceat(l, starts),
            "c": minutes["c"].to_numpy(dtype=float)[ends],
            "minutes": ends - starts + 1,
            "i0": starts,
            "i1": ends,
        }
    )
    return bars, bar_index
