from pathlib import Path

import pandas as pd
import pytest

from strat_enrichment import enrich

SAMPLE = Path(__file__).resolve().parent.parent / "sample" / "futures_1m.parquet"


@pytest.fixture(scope="session")
def raw():
    return pd.read_parquet(SAMPLE)


@pytest.fixture(scope="session")
def out(raw):
    return enrich(raw, market="futures", tfc_sets={"all": ["15m", "30m", "1h", "4h", "D", "W", "M"], "htf": ["D", "W", "M"]})
