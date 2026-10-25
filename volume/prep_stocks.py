"""Turn 1-minute stock/ETF bars into the volume study's stocks input.

    python volume/prep_stocks.py OUT.parquet FILE [FILE ...] [--volume-from VFILE ...] [--start D] [--end D]

FILE: any 1-minute bars with a timestamp (bar start) and open, high, low, close, volume. Reads Databento
ohlcv-1m (.dbn / .dbn.zst with `pip install databento`, or its .parquet / .csv: ts_event, open, ...), or a
.csv / .parquet with columns timestamp|ts|time, open, high, low, close, volume. Naive timestamps are read as UTC.

--volume-from: take prices from FILE and volume from VFILE, matched by minute. The report's SPY used
consolidated prices with single-venue (NYSE Arca, Databento ARCX.PILLAR) volume; relative volume is a ratio, so
one venue's share mostly cancels. Using one source for both is fine and will give slightly different numbers.

--splits: back-adjust prices and volume for stock splits, so bar types, ranges and relative volume line up
across the split (the ratios a split changes cancel).

Output columns: ts (bar start, UTC epoch seconds), o, h, l, c, v. Only 09:30-16:00 ET is used by the study."""
import argparse
import numpy as np
import pandas as pd


def read_any(path):
    if path.endswith((".dbn", ".dbn.zst")):
        import databento as db
        df = db.DBNStore.from_file(path).to_df().reset_index()
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
        if not any(c in df.columns for c in ("ts_event", "timestamp", "ts", "time")):
            df = df.reset_index()
    tc = next(c for c in ("ts_event", "timestamp", "ts", "time", "index") if c in df.columns)
    t = df[tc]
    if pd.api.types.is_integer_dtype(t):
        unit = "ns" if t.max() > 1e15 else ("ms" if t.max() > 1e11 else "s")
        t = pd.to_datetime(t, unit=unit, utc=True)
    else:
        t = pd.to_datetime(t, utc=True)
    cols = {c.lower(): c for c in df.columns}
    g = lambda k: df[cols[k]].astype(float)
    return pd.DataFrame({"ts": (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s"),
                         "o": g("open"), "h": g("high"), "l": g("low"), "c": g("close"), "v": g("volume")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("files", nargs="+")
    ap.add_argument("--volume-from", nargs="+")
    ap.add_argument("--start"); ap.add_argument("--end", help="inclusive date, ET")
    ap.add_argument("--splits", help="split-adjust: comma list of FIRST_SPLIT_ADJUSTED_SESSION:ratio, e.g. "
                                     "2021-07-20:4,2024-06-10:10. Earlier bars get prices / ratio and volume x ratio.")
    a = ap.parse_args()
    d = pd.concat([read_any(f) for f in a.files]).drop_duplicates("ts")
    if a.volume_from:
        v = pd.concat([read_any(f) for f in a.volume_from]).drop_duplicates("ts")
        d = d.drop(columns="v").merge(v[["ts", "v"]], on="ts", how="left")
        print(f"volume matched on {d.v.notna().mean():.1%} of minutes")
    d = d.sort_values("ts").reset_index(drop=True)
    if a.splits:
        et = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
        for item in a.splits.split(","):
            day, r = item.split(":")
            before = (et < pd.Timestamp(day, tz="America/New_York")).values
            d.loc[before, ["o", "h", "l", "c"]] /= float(r)
            d.loc[before, "v"] *= float(r)
            print(f"split {day} x{r}: adjusted {before.sum():,} earlier minutes")
    t = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    mod = t.dt.hour * 60 + t.dt.minute
    keep = (mod >= 570) & (mod < 960)
    if a.start:
        keep &= t >= pd.Timestamp(a.start, tz="America/New_York")
    if a.end:
        keep &= t < pd.Timestamp(a.end, tz="America/New_York") + pd.Timedelta(days=1)
    d = d[keep.values]
    d.to_parquet(a.out, index=False)
    print(f"{a.out}: {len(d):,} minutes, {t[keep].dt.date.nunique()} sessions")


if __name__ == "__main__":
    main()
