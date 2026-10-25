"""Smoke test for volume/: the study runs end to end on synthetic bars (random walk, random volume)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strat_enrichment.sample import synthetic_minutes

ROOT = Path(__file__).resolve().parents[1]


def with_volume(df, seed):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["v"] = rng.gamma(2.0, 500.0, len(df)).round()
    return df


def test_volume_study_runs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fut = with_volume(synthetic_minutes("futures", start="2022-09-04", end="2023-03-04"), 1)
    fut["contract"] = "SYN"
    fut.to_parquet(data / "nq.parquet")
    stk = with_volume(synthetic_minutes("stocks", start="2022-09-04", end="2023-03-04"), 2)
    stk.drop(columns=[c for c in ("contract",) if c in stk], errors="ignore").to_parquet(data / "spy.parquet")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "volume" / "run.py"), str(data), str(out)],
                       capture_output=True, text=True, check=True)
    assert "RESULTS.md: ok" in r.stdout, r.stdout + r.stderr
    s1 = pd.read_parquet(out / "work" / "s1_nq.parquet")
    assert len(s1) > 100 and s1.R.notna().all()
    assert (out / "RESULTS.md").read_text().count("| nq | H") >= 9


def test_volume_study_crypto_runs(tmp_path):
    rng = np.random.default_rng(3)
    ts = pd.date_range("2022-10-01", "2023-03-01", freq="1min", tz="UTC", inclusive="left")
    c = 20000 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(ts))))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.0004, len(ts))) * c
    df = pd.DataFrame({"ts": (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s"), "o": o, "h": np.maximum(o, c) + w, "l": np.minimum(o, c) - w, "c": c,
                       "v": rng.gamma(2.0, 5.0, len(ts))})
    data = tmp_path / "data"
    data.mkdir()
    df.to_parquet(data / "btc.parquet")
    df.to_parquet(data / "btccb.parquet")
    work = tmp_path / "work"
    for name in ("btc", "btccb"):
        subprocess.run([sys.executable, str(ROOT / "volume" / "volstudy.py"), str(data / f"{name}.parquet"), name, str(work),
                        "--market", "crypto"], check=True, capture_output=True)
    r = subprocess.run([sys.executable, str(ROOT / "volume" / "analyze_btc.py"), str(work), str(tmp_path / "R.md")],
                       capture_output=True, text=True, cwd=ROOT / "volume")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Direction:" in (tmp_path / "R.md").read_text()


def test_volume_study_stocks_pooled_runs(tmp_path):
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    for i, n in enumerate(("aaa", "bbb")):
        stk = with_volume(synthetic_minutes("stocks", start="2022-09-04", end="2023-03-04"), 10 + i)
        stk[["ts", "o", "h", "l", "c", "v"]].to_parquet(data / f"{n}.parquet")
        subprocess.run([sys.executable, str(ROOT / "volume" / "volstudy.py"), str(data / f"{n}.parquet"), n, str(work),
                        "--market", "stocks", "--cost-pct", "0.0002"], capture_output=True, text=True, check=True)
    r = subprocess.run([sys.executable, str(ROOT / "volume" / "analyze_stocks.py"), str(work), str(tmp_path / "R.md"), "aaa", "bbb"],
                       capture_output=True, text=True, cwd=ROOT / "volume")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "Direction:" in (tmp_path / "R.md").read_text()
