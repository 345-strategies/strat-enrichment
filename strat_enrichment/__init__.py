"""Rebuild TheStrat bar states on every timeframe from 1-minute bars.

    >>> import pandas as pd
    >>> from strat_enrichment import enrich
    >>> out = enrich(pd.read_parquet("sample/futures_1m.parquet"), market="futures")
    >>> out.bars[out.bars.tf == "D"][["session", "type", "combo3", "tfc_all"]].tail(3)

Definitions come from thestrat_grammar; this package builds the bars and
decides which candle is compared with which.
"""

from .bars import TIMEFRAMES, build_bars
from .pipeline import Enriched, enrich
from .rolls import back_adjust
from .sessions import MARKETS, sessionize
from .states import classify_bars, forming_states, tfc

__version__ = "0.1.0"

__all__ = [
    "TIMEFRAMES",
    "MARKETS",
    "Enriched",
    "enrich",
    "back_adjust",
    "sessionize",
    "build_bars",
    "classify_bars",
    "forming_states",
    "tfc",
]
