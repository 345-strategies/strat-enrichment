"""Futures roll back-adjustment.

A continuous futures series switches contract at each roll. The new contract
trades at a different price than the old one, so without adjustment the roll
prints a jump that the classifier reads as a 2 or a 3 that never happened.

At each change of `contract`, the gap is the new contract's first open minus
the old contract's last close. Every bar before the switch is shifted by that
gap. The latest contract stays at its real prices and history is moved to meet
it.

The gap is measured between the two minutes either side of the switch, so roll
in the middle of a session. A switch across a weekend would also remove the
weekend's real move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def back_adjust(minutes: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return (adjusted minutes, rolls).

    `minutes` must be sorted by `ts` and carry raw prices of `contract`. The
    result keeps `contract` and adds `adj`, the offset added to each row, so
    raw price = adjusted price - adj.
    """
    contract = minutes["contract"].to_numpy()
    o = minutes["o"].to_numpy(dtype=float)
    c = minutes["c"].to_numpy(dtype=float)

    switches = np.flatnonzero(contract[1:] != contract[:-1]) + 1
    gaps = o[switches] - c[switches - 1]

    # Every bar before a switch carries that switch's gap, so the oldest bars
    # carry the sum of all the gaps after them.
    adj = np.zeros(len(minutes))
    for i, gap in zip(switches, gaps):
        adj[:i] += gap

    out = minutes.copy()
    for col in ("o", "h", "l", "c"):
        out[col] = minutes[col].to_numpy(dtype=float) + adj
    out["adj"] = adj

    rolls = [
        {
            "ts": int(minutes["ts"].iat[i]),
            "old": str(contract[i - 1]),
            "new": str(contract[i]),
            "gap": float(gap),
        }
        for i, gap in zip(switches, gaps)
    ]
    return out, rolls
