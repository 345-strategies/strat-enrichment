"""Every output state matches thestrat_grammar, recomputed from scratch."""

import numpy as np
import pandas as pd
from thestrat_grammar import Candle, Structure, classify, combo, continuity

ET = "America/New_York"


def candle(row):
    return Candle(row.o, row.h, row.l, row.c)


def test_completed_bars_match_grammar(out):
    for tf, b in out.bars.groupby("tf"):
        rows = list(b.itertuples())
        for i in range(1, len(rows)):
            s = classify(candle(rows[i]), candle(rows[i - 1]))
            r = rows[i]
            assert r.type == s.notation(), (tf, r.session)
            assert r.above_open == s.above_open
            assert r.took_c1h == (s.structure in (Structure.TWO_UP, Structure.OUTSIDE))
            assert r.took_c1l == (s.structure in (Structure.TWO_DOWN, Structure.OUTSIDE))
            if i >= 2:
                s1 = classify(candle(rows[i - 1]), candle(rows[i - 2]))
                assert r.combo2 == combo([s1, s])
            if i >= 3:
                s2 = classify(candle(rows[i - 2]), candle(rows[i - 3]))
                assert r.combo3 == combo([s2, s1, s])
        assert rows[0].type == ""


def test_forming_states_match_grammar(out):
    """Rebuild the forming bar from the raw minutes at random moments."""
    m = out.minutes
    rng = np.random.default_rng(0)
    for tf, b in out.bars.groupby("tf"):
        starts = b.ts.to_numpy()
        for i in rng.choice(len(m), 300, replace=False):
            ts = m.ts.iat[i]
            j = np.searchsorted(starts, ts, side="right") - 1
            if j < 1 or j >= len(b) or ts > b.ts_last.iat[j]:
                continue  # first bar has no C1; past the last completed bar
            part = m[(m.ts >= starts[j]) & (m.ts <= ts)]
            forming = Candle(part.o.iat[0], part.h.max(), part.l.min(), part.c.iat[-1])
            s = classify(forming, candle(b.iloc[j - 1]), live=True)
            assert m[f"{tf}_type"].iat[i] == s.notation(), (tf, ts)
            assert m[f"{tf}_above_open"].iat[i] == s.above_open


def test_tfc_matches_grammar(out):
    m = out.minutes
    rng = np.random.default_rng(1)
    for i in rng.choice(len(m), 2000, replace=False):
        for name, tfs in {"all": ["15m", "30m", "1h", "4h", "D", "W", "M"], "htf": ["D", "W", "M"]}.items():
            signs = [bool(m[f"{tf}_above_open"].iat[i]) for tf in tfs]
            expected = continuity(Candle(0, 1, 0, 1 if up else 0) for up in signs).value
            assert m[f"tfc_{name}"].iat[i] == expected


def test_bar_tfc_is_the_reading_at_its_close(out):
    m = out.minutes.set_index("ts")
    for _, b in out.bars.groupby("tf"):
        assert (b.tfc_all.to_numpy() == m.loc[b.ts_last, "tfc_all"].to_numpy()).all()
