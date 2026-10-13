"""The README quickstart, run as a user would."""

import json
import subprocess
import sys

import pandas as pd

from conftest import SAMPLE


def test_cli_on_the_sample(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "strat_enrichment", str(SAMPLE), str(tmp_path), "--tfc", "htf=D,W,M", "--tfc", "intraday=15m,1h,4h"],
        capture_output=True, text=True, check=True,
    )
    assert "wrote" in r.stdout
    bars = pd.read_parquet(tmp_path / "bars.parquet")
    minutes = pd.read_parquet(tmp_path / "minutes.parquet")
    assert set(bars.tf) == {"15m", "30m", "1h", "4h", "D", "W", "M"}
    assert {"tfc_htf", "tfc_intraday"} <= set(bars.columns) and "tfc_all" not in bars.columns
    assert len(minutes) == len(pd.read_parquet(SAMPLE))
    assert json.loads((tmp_path / "rolls.json").read_text())[0]["new"] == "SYNM6"
