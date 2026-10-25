"""Turn 1-minute crypto bars into the volume study's crypto input (sessions are UTC days).

    python volume/prep_crypto.py OUT.parquet FILE [FILE ...] [--start 2020-01-01] [--end 2026-05-04]

FILE: Binance kline zips or CSVs from data.binance.vision (no header: open time, open, high, low, close, volume,
...; open time in milliseconds, or microseconds from 2025 on), or any .csv / .parquet with a time column
(time|timestamp|ts|bar|open_time) and open, high, low, close, volume. Naive times are read as UTC.
Volume is in the base asset (BTC). Output columns: ts (bar start, UTC epoch seconds), o, h, l, c, v."""
import argparse, io, zipfile
import numpy as np
import pandas as pd

KL = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
      "taker_base", "taker_quote", "ignore"]


def read_any(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            raw = z.read(z.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(raw), header=None)
        if not str(df.iloc[0, 0]).strip().lstrip("-").isdigit():    # a few archive files carry a header row
            df = df.iloc[1:]
        df = df.iloc[:, :12]
        df.columns = KL[:df.shape[1]]
        t = df["open_time"].astype("int64")
        ts = np.where(t > 10**14, t // 10**6, t // 10**3)          # microseconds (2025+) or milliseconds
        return pd.DataFrame({"ts": ts, "o": df.open.astype(float), "h": df.high.astype(float),
                             "l": df.low.astype(float), "c": df.close.astype(float), "v": df.volume.astype(float)})
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    if not any(c in df.columns for c in ("time", "timestamp", "ts", "bar", "open_time")):
        df = df.reset_index()
    tc = next(c for c in ("time", "timestamp", "ts", "bar", "open_time", "index") if c in df.columns)
    t = df[tc]
    if pd.api.types.is_integer_dtype(t):
        unit = "ns" if t.max() > 1e17 else ("us" if t.max() > 1e14 else ("ms" if t.max() > 1e11 else "s"))
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
    ap.add_argument("--start"); ap.add_argument("--end", help="inclusive UTC date")
    a = ap.parse_args()
    d = pd.concat([read_any(f) for f in a.files]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    t = pd.to_datetime(d["ts"], unit="s", utc=True)
    keep = np.ones(len(d), bool)
    if a.start:
        keep &= (t >= pd.Timestamp(a.start, tz="UTC")).values
    if a.end:
        keep &= (t < pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)).values
    d = d[keep]
    d.to_parquet(a.out, index=False)
    print(f"{a.out}: {len(d):,} minutes, {t[keep].dt.date.nunique()} UTC days")


if __name__ == "__main__":
    main()
