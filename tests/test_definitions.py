"""The canonical edge cases, end to end through the pipeline.

Each case is a few 1-minute bars inside one futures session, built on the 1m
timeframe so every minute is its own bar.
"""

import pandas as pd
import pytest

from strat_enrichment import enrich

T0 = int(pd.Timestamp("2026-01-06 10:00", tz="America/New_York").timestamp())


def run(*ohlc):
    m = pd.DataFrame(ohlc, columns=["o", "h", "l", "c"]).assign(ts=[T0 + 60 * i for i in range(len(ohlc))])
    m = pd.concat([m, m.tail(1).assign(ts=T0 + 60 * len(ohlc))])  # the last bar is dropped as incomplete
    return enrich(m, timeframes=["1m"]).bars


C1 = (100, 110, 90, 105)


@pytest.mark.parametrize(
    "bar,expected",
    [
        ((105, 110, 95, 108), "1u"),     # equal high is not a break
        ((105, 109, 90, 95), "1d"),      # equal low is not a break
        ((105, 111, 95, 111), "2u"),     # took the high and held it
        ((105, 111, 95, 110), "F2u"),    # closed back at the prior high: inside, edges included
        ((95, 105, 89, 90), "F2d"),      # closed back at the prior low
        ((105, 111, 89, 106), "3u"),     # outside, closed above its own open, inside the prior range: no F3
        ((112, 115, 89, 111), "3d"),     # outside, closed below its own open, above the prior high
        ((100, 105, 95, 100), "1d"),     # close at the open is not above the open
    ],
)
def test_bar_types(bar, expected):
    b = run(C1, bar)
    assert b.type.iat[1] == expected


def test_doji_is_below_open_and_took_flags_follow_structure():
    b = run(C1, (105, 111, 89, 105))
    assert b.type.iat[1] == "3d" and not b.above_open.iat[1]
    assert b.took_c1h.iat[1] and b.took_c1l.iat[1]


def test_only_twos_fail():
    b = run(C1, (105, 111, 89, 100), (100, 100, 100, 100))
    assert b.type.tolist() == ["", "3d", "1d"]


def test_combos_drop_the_sign_of_an_interior_inside_bar():
    b = run(C1, (95, 112, 95, 111), (108, 111, 100, 109), (109, 115, 104, 114))
    assert b.combo2.tolist() == ["", "", "2u-1u", "1u-2u"]
    assert b.combo3.tolist() == ["", "", "", "2u-1-2u"]


def test_a_gap_gets_no_type_of_its_own():
    b = run(C1, (120, 125, 118, 119))
    assert b.type.iat[1] == "2u" and not b.above_open.iat[1]
