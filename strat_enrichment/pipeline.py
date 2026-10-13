"""1-minute bars in, enriched bars and live states out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bars import TIMEFRAMES, build_bars, tf_minutes
from .rolls import back_adjust
from .sessions import sessionize
from .states import classify_bars, forming_states, tfc

STATE_COLUMNS = ("type", "above_open", "took_c1h", "took_c1l")


@dataclass
class Enriched:
    bars: pd.DataFrame        # completed bars, every timeframe, stacked with a `tf` column
    minutes: pd.DataFrame     # every 1-minute bar with each timeframe's forming state
    rolls: list[dict] = field(default_factory=list)


def enrich(
    minutes: pd.DataFrame,
    market: str = "futures",
    timeframes: Sequence[str] = TIMEFRAMES,
    tfc_sets: Mapping[str, Sequence[str]] | None = None,
) -> Enriched:
    """Build and classify every timeframe from 1-minute bars.

    `minutes` columns: `ts` (bar start, UTC epoch seconds), `o`, `h`, `l`, `c`.
    With a `contract` column the prices are treated as raw contract prices and
    back-adjusted at each roll. `tfc_sets` names the timeframe sets to read
    continuity across; the default is one set, `all`, of every timeframe.
    """
    timeframes = list(timeframes)
    for tf in timeframes:
        tf_minutes(tf)
    tfc_sets = dict(tfc_sets or {"all": timeframes})
    for name, tfs in tfc_sets.items():
        missing = sorted(set(tfs) - set(timeframes))
        if missing:
            raise ValueError(f"TFC set {name!r} uses timeframes that are not built: {missing}")

    m = _validated(minutes)
    rolls: list[dict] = []
    if "contract" in m.columns:
        m, rolls = back_adjust(m)
    m = sessionize(m, market)
    if m.empty:
        raise ValueError(f"no minutes fall inside {market} session hours")

    all_bars, live = [], {}
    for tf in timeframes:
        bars, bar_index = build_bars(m, tf)
        bars = pd.concat([bars, classify_bars(bars)], axis=1)
        states = forming_states(m, bars, bar_index)
        closing = states["type"].to_numpy()[bars["i1"].to_numpy()]
        bad = np.flatnonzero(closing != bars["type"].to_numpy())
        if len(bad):
            raise RuntimeError(f"{tf}: forming state at bar close differs from the completed bar at bar {bad[0]}")
        bars.insert(0, "tf", tf)
        all_bars.append(bars)
        live[tf] = states

    out_min = m[["ts", "session", "o", "h", "l", "c"]].copy()
    for col in ("contract", "adj"):
        if col in m.columns:
            out_min[col] = m[col].to_numpy()
    for tf in timeframes:
        for col in STATE_COLUMNS:
            out_min[f"{tf}_{col}"] = live[tf][col].to_numpy()
    for name, tfs in tfc_sets.items():
        labels = tfc(np.column_stack([live[tf]["above_open"].to_numpy() for tf in tfs]))
        out_min[f"tfc_{name}"] = labels
        for bars in all_bars:
            bars[f"tfc_{name}"] = labels[bars["i1"].to_numpy()]

    # The file can end before the last bar of a timeframe closes, so that bar is
    # left out of the completed bars. Its live state is in the minutes.
    out_bars = pd.concat([b.iloc[:-1] for b in all_bars], ignore_index=True)
    out_bars = out_bars.drop(columns=["i0", "i1"])
    out_bars["session"] = _iso(out_bars["session"])
    out_min["session"] = _iso(out_min["session"])
    return Enriched(bars=out_bars, minutes=out_min.reset_index(drop=True), rolls=rolls)


def _validated(minutes: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("ts", "o", "h", "l", "c") if c not in minutes.columns]
    if missing:
        raise ValueError(f"1-minute input is missing columns: {missing}")
    m = minutes.sort_values("ts", kind="stable").reset_index(drop=True)
    if m["ts"].duplicated().any():
        raise ValueError("1-minute input has duplicate ts values")
    if m[["o", "h", "l", "c"]].isna().any().any():
        raise ValueError("1-minute input has missing prices")
    bad = (m["h"] < m[["o", "c", "l"]].max(axis=1)) | (m["l"] > m[["o", "c"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} 1-minute bars have open or close outside high-low")
    m["ts"] = m["ts"].astype(np.int64)
    return m


def _iso(session: pd.Series) -> np.ndarray:
    return np.datetime_as_string(session.to_numpy().astype("datetime64[D]"), unit="D")
