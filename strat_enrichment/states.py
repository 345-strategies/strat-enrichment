"""Strat states, taken from the public grammar.

Every definition here comes from `thestrat_grammar` (the grammar/ package in
345-strategies/thestrat-suite): bar types, the sign channel, failed 2s,
combos and continuity. This module only decides which candle is compared with
which; it never re-derives a definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from thestrat_grammar import Candle, State, Structure, classify, combo, continuity

TOOK_HIGH = (Structure.TWO_UP, Structure.OUTSIDE)
TOOK_LOW = (Structure.TWO_DOWN, Structure.OUTSIDE)


def _candles(o, h, l, c) -> list[Candle]:
    return [Candle(*x) for x in zip(o.tolist(), h.tolist(), l.tolist(), c.tolist())]


def _took(states: list[State | None], sides: tuple[Structure, ...]) -> pd.array:
    return pd.array([None if s is None else s.structure in sides for s in states], dtype="boolean")


def classify_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Type, sign, took-C1 flags and combos for each completed bar.

    Each bar is classified against the bar before it. The first bar has no C1,
    so its type is empty and its took flags are missing.
    """
    candles = _candles(bars["o"], bars["h"], bars["l"], bars["c"])
    states: list[State | None] = [None] + [classify(cur, c1) for c1, cur in zip(candles, candles[1:])]
    combo2 = [
        combo([s[1], s[0]]) if s[0] and s[1] else ""
        for s in zip(states, [None] + states[:-1])
    ]
    combo3 = [
        combo([s[2], s[1], s[0]]) if s[0] and s[1] and s[2] else ""
        for s in zip(states, [None] + states[:-1], [None, None] + states[:-2])
    ]
    return pd.DataFrame(
        {
            "type": [s.notation() if s else "" for s in states],
            "above_open": [x.above_open for x in candles],
            "took_c1h": _took(states, TOOK_HIGH),
            "took_c1l": _took(states, TOOK_LOW),
            "combo2": combo2,
            "combo3": combo3,
        },
        index=bars.index,
    )


def forming_states(minutes: pd.DataFrame, bars: pd.DataFrame, bar_index: np.ndarray) -> pd.DataFrame:
    """The forming bar's live state at the close of every 1-minute bar.

    At each minute the forming bar is its timeframe bar so far: the bar's open,
    the running high and low, and this minute's close. It is classified against
    the last completed bar (C1).
    """
    grouped = pd.Series(minutes["h"].to_numpy(), copy=False).groupby(bar_index)
    run_h = grouped.cummax().to_numpy()
    run_l = pd.Series(minutes["l"].to_numpy(), copy=False).groupby(bar_index).cummin().to_numpy()
    forming = _candles(bars["o"].to_numpy()[bar_index], run_h, run_l, minutes["c"].to_numpy())

    priors = [None] + _candles(bars["o"], bars["h"], bars["l"], bars["c"])
    states = [
        classify(cur, c1, live=True) if c1 is not None else None
        for cur, c1 in zip(forming, (priors[j] for j in bar_index.tolist()))
    ]
    return pd.DataFrame(
        {
            "type": [s.notation() if s else "" for s in states],
            "above_open": [x.above_open for x in forming],
            "took_c1h": _took(states, TOOK_HIGH),
            "took_c1l": _took(states, TOOK_LOW),
        }
    )


def tfc(signs: np.ndarray) -> np.ndarray:
    """Full Timeframe Continuity for each row of a (rows x timeframes) sign matrix.

    Continuity reads only the sign channel, so the grammar is evaluated once per
    distinct pattern of signs and the answer is mapped back to every row.
    """
    patterns, inverse = np.unique(signs, axis=0, return_inverse=True)
    labels = np.array(
        [continuity(Candle(0.0, 1.0, 0.0, 1.0 if up else 0.0) for up in row).value for row in patterns]
    )
    return labels[inverse.reshape(-1)]
