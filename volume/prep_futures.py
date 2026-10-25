"""Turn Databento CME 1-minute bars into the volume study's futures input: one contract per minute, raw
(unadjusted) prices, with volume.

    python volume/prep_futures.py NQ OUT.parquet FILE [FILE ...] [--start 2018-01-01] [--end 2026-09-22]

FILE: Databento GLBX.MDP3 ohlcv-1m, as .dbn / .dbn.zst (needs `pip install databento`), .parquet or .csv, with
the per-contract raw symbols (e.g. NQH8, NQM8 ...). A parent-symbology pull (symbols="NQ.FUT", stype_in="parent")
returns every contract plus spreads; spreads and serial months are dropped here.

Roll rule (the one in the report's data): hold the quarterly contract until 00:00 ET on the Tuesday of its
expiration week (third Friday of Mar/Jun/Sep/Dec, minus 3 days), then switch to the next one. That is late, when
the expiring contract's volume has already moved: the study drops the 3 sessions before each switch from every
signal and volume baseline, and you should not loosen that.

Output columns: ts (bar start, UTC epoch seconds), o, h, l, c, v, contract."""
import argparse, re
import numpy as np
import pandas as pd

MONTHS = "HMUZ"


def read_any(path):
    if path.endswith((".dbn", ".dbn.zst")):
        import databento as db
        df = db.DBNStore.from_file(path).to_df()
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if "ts_event" not in df.columns:
        df = df.reset_index()
    t = df["ts_event"]
    if pd.api.types.is_integer_dtype(t):
        t = pd.to_datetime(t, unit="ns", utc=True)
    else:
        t = pd.to_datetime(t, utc=True)
    out = pd.DataFrame({"ts": (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s"),
                        "o": df["open"].astype(float), "h": df["high"].astype(float), "l": df["low"].astype(float),
                        "c": df["close"].astype(float), "v": df["volume"].astype(float), "symbol": df["symbol"].astype(str)})
    return out


def roll_times(first_year, last_year):
    """[(roll time UTC epoch s, contract year, month letter)], the moment each quarterly stops being front."""
    out = []
    for y in range(first_year - 1, last_year + 2):
        for i, m in enumerate((3, 6, 9, 12)):
            d = pd.Timestamp(year=y, month=m, day=1)
            fri3 = d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)
            roll = (fri3 - pd.Timedelta(days=3)).tz_localize("America/New_York")
            out.append((int(roll.tz_convert("UTC").timestamp()), y, MONTHS[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="product root, e.g. NQ or ES")
    ap.add_argument("out")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--start"); ap.add_argument("--end", help="inclusive date, ET")
    a = ap.parse_args()
    d = pd.concat([read_any(f) for f in a.files], ignore_index=True)
    pat = re.compile(rf"^{a.root}([{MONTHS}])(\d{{1,2}})$")
    m = d["symbol"].str.extract(pat)
    d = d[m[0].notna()].copy()
    d["mon"], d["yd"] = m[0][m[0].notna()], m[1][m[0].notna()]
    years = pd.to_datetime(d["ts"], unit="s", utc=True).dt.year
    rolls = roll_times(int(years.min()), int(years.max()))
    edges = np.array([r[0] for r in rolls])
    k = np.searchsorted(edges, d["ts"].values, side="right")      # first roll after this minute = front contract
    fy = np.array([rolls[i][1] for i in k]); fm = np.array([rolls[i][2] for i in k])
    yd = d["yd"].astype(int).values
    full = np.where(d["yd"].str.len() == 1, (fy // 10) * 10 + yd, 2000 + yd)
    full = np.where(full - fy > 5, full - 10, full)                # one-digit years across a decade turn
    d = d[(full == fy) & (d["mon"].values == fm)]
    d = d.rename(columns={"symbol": "contract"})[["ts", "o", "h", "l", "c", "v", "contract"]]
    d = d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    t = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    if a.start:
        d = d[t >= pd.Timestamp(a.start, tz="America/New_York")]
    if a.end:
        d = d[t < pd.Timestamp(a.end, tz="America/New_York") + pd.Timedelta(days=1)]
    d.to_parquet(a.out, index=False)
    print(f"{a.out}: {len(d):,} minutes, {d.contract.nunique()} contracts, "
          f"{pd.to_datetime(d.ts.min(), unit='s')} .. {pd.to_datetime(d.ts.max(), unit='s')} UTC")


if __name__ == "__main__":
    main()
