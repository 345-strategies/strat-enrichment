"""Download Binance spot 1-minute klines from the public archive (data.binance.vision, no account needed).

    python volume/fetch_binance.py BTCUSDT 2020-01 2026-09 raw/binance/

Saves one monthly zip per month into the folder (daily zips for the current month, which has no monthly file
yet). Then: python volume/prep_crypto.py data/btc.parquet raw/binance/*.zip"""
import os, sys, urllib.request
import pandas as pd

BASE = "https://data.binance.vision/data/spot"


def get(url, path):
    if os.path.exists(path):
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
        return True
    except Exception as e:
        print("missing", url, e)
        return False


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    sym, start, end, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    for m in pd.period_range(start, end, freq="M"):
        name = f"{sym}-1m-{m.year}-{m.month:02d}"
        if not get(f"{BASE}/monthly/klines/{sym}/1m/{name}.zip", os.path.join(out, name + ".zip")):
            for d in pd.date_range(m.start_time, m.end_time.normalize(), freq="D"):
                dn = f"{sym}-1m-{d:%Y-%m-%d}"
                get(f"{BASE}/daily/klines/{sym}/1m/{dn}.zip", os.path.join(out, dn + ".zip"))
    print("done", out)


if __name__ == "__main__":
    main()
