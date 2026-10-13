"""Bar building: session anchors, aggregation, rolls, stocks hours."""

import numpy as np
import pandas as pd
import pytest

from strat_enrichment import enrich
from strat_enrichment.sample import CARRY, ROLL_ET, synthetic_minutes

ET = "America/New_York"


def et(ts):
    return pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(ET)


@pytest.mark.parametrize("tf,rule,offset", [("15m", "15min", None), ("1h", "1h", None), ("4h", "4h", "2h")])
def test_intraday_bars_match_a_plain_resample(out, tf, rule, offset):
    """On futures, session-anchored 15m and 1h are clock-aligned and 4h starts at 02, 06, 10, 14, 18, 22 ET."""
    m = out.minutes.assign(t=et(out.minutes.ts).dt.tz_localize(None)).set_index("t")
    r = m.resample(rule, offset=offset).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    b = out.bars[out.bars.tf == tf]
    r = r.loc[et(b.ts).dt.floor(rule if offset is None else "1h").dt.tz_localize(None)]
    assert np.allclose(r[["o", "h", "l", "c"]].to_numpy(), b[["o", "h", "l", "c"]].to_numpy())


def test_sessions_open_at_18_et_across_dst(out):
    d = out.bars[out.bars.tf == "D"]
    starts = et(d.ts)
    assert (starts.dt.strftime("%H:%M") == "18:00").all()
    # Sunday evening opens Monday's session; the session is dated by the day it closes
    sunday = starts.dt.dayofweek == 6
    assert (pd.to_datetime(d.session[sunday]).dt.dayofweek == 0).all()
    assert {o.utcoffset().total_seconds() / 3600 for o in starts} == {-5.0, -4.0}


def test_week_and_month_follow_the_session_date(out):
    w = out.bars[out.bars.tf == "W"]
    assert (et(w.ts).dt.dayofweek == 6).all()            # each week opens Sunday 18:00 ET
    m = out.bars[out.bars.tf == "M"]
    first = pd.to_datetime(m.session)
    assert (first.dt.day <= 3).all()                     # first session of the month
    assert (first.dt.month.diff().dropna() % 12 == 1).all()


def test_roll_gap_is_removed_and_latest_contract_is_real(raw, out):
    m = out.minutes
    at = int(pd.Timestamp(ROLL_ET, tz=ET).timestamp())
    i = int(np.flatnonzero(m.ts.to_numpy() == at)[0])
    assert m.contract.iat[i - 1] == "SYNH6" and m.contract.iat[i] == "SYNM6"
    assert m.o.iat[i] == m.c.iat[i - 1]                  # no jump at the switch
    assert (m.adj.iloc[:i] == CARRY).all() and (m.adj.iloc[i:] == 0).all()
    tail = raw.set_index("ts").loc[m.ts.iloc[i:]]
    assert np.array_equal(tail.c.to_numpy(), m.c.iloc[i:].to_numpy())
    assert len(out.rolls) == 1 and out.rolls[0]["gap"] == CARRY


def test_last_bar_is_left_out_as_possibly_incomplete(out):
    for tf, b in out.bars.groupby("tf"):
        assert b.ts_last.max() < out.minutes.ts.max()


def test_stocks_regular_hours_only():
    s = synthetic_minutes("stocks", start="2026-01-05", end="2026-01-17")
    extra = pd.DataFrame({"ts": [int(pd.Timestamp(t, tz=ET).timestamp()) for t in ("2026-01-05 08:00", "2026-01-05 16:30")],
                          "o": 500.0, "h": 500.0, "l": 500.0, "c": 500.0})
    out = enrich(pd.concat([s, extra]), market="stocks")
    clock = et(out.minutes.ts).dt.strftime("%H:%M")
    assert clock.min() == "09:30" and clock.max() == "15:59"
    h1 = out.bars[out.bars.tf == "1h"]
    assert set(et(h1.ts).dt.strftime("%H:%M")) == {"09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"}
    assert set(h1.minutes[et(h1.ts).dt.strftime("%H:%M") == "15:30"]) == {30}
    h4 = out.bars[out.bars.tf == "4h"]
    assert set(et(h4.ts).dt.strftime("%H:%M")) == {"09:30", "13:30"}


def test_a_session_through_the_dst_change_is_refused():
    # 2026-11-01 01:00-01:59 ET happens twice; a 24-hour feed would merge them into one 1h bar
    ts = np.arange(pd.Timestamp("2026-11-01 04:00", tz="UTC").timestamp(), pd.Timestamp("2026-11-01 08:00", tz="UTC").timestamp(), 60).astype(int)
    m = pd.DataFrame({"ts": ts, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0})
    with pytest.raises(ValueError, match="daylight saving"):
        enrich(m, market="futures", timeframes=["1h"])


def test_unknown_timeframe_in_a_tfc_set_is_refused(raw):
    with pytest.raises(ValueError, match="not built"):
        enrich(raw.head(5000), timeframes=["15m", "1h"], tfc_sets={"x": ["15m", "D"]})
