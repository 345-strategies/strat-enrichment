"""strat-enrich: 1-minute parquet in, enriched parquet out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .bars import TIMEFRAMES
from .pipeline import enrich
from .sessions import MARKETS


def _tfc_set(text: str) -> tuple[str, list[str]]:
    name, sep, tfs = text.partition("=")
    if not sep or not name or not tfs:
        raise argparse.ArgumentTypeError(f"expected NAME=TF,TF,... got {text!r}")
    return name, tfs.split(",")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="strat-enrich",
        description="Rebuild Strat bar states on every timeframe from 1-minute bars.",
    )
    p.add_argument("input", type=Path, help="1-minute parquet: ts (UTC epoch seconds, bar start), o, h, l, c; futures add contract")
    p.add_argument("outdir", type=Path, help="folder for bars.parquet, minutes.parquet and rolls.json")
    p.add_argument("--market", choices=sorted(MARKETS), default="futures",
                   help="futures: sessions open 18:00 ET; stocks: 09:30-16:00 ET (default: futures)")
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES),
                   help=f"comma-separated, Nm/Nh/D/W/M (default: {','.join(TIMEFRAMES)})")
    p.add_argument("--tfc", type=_tfc_set, action="append", metavar="NAME=TF,TF",
                   help="a named timeframe set to read continuity across; repeatable (default: all=every timeframe)")
    a = p.parse_args(argv)

    out = enrich(
        pd.read_parquet(a.input),
        market=a.market,
        timeframes=a.timeframes.split(","),
        tfc_sets=dict(a.tfc) if a.tfc else None,
    )
    a.outdir.mkdir(parents=True, exist_ok=True)
    out.bars.to_parquet(a.outdir / "bars.parquet", index=False)
    out.minutes.to_parquet(a.outdir / "minutes.parquet", index=False)
    if out.rolls:
        (a.outdir / "rolls.json").write_text(json.dumps(out.rolls, indent=1) + "\n")

    print(f"{len(out.minutes):,} minutes, {out.minutes.session.iat[0]} to {out.minutes.session.iat[-1]}, "
          f"{len(out.rolls)} rolls")
    for tf, b in out.bars.groupby("tf", sort=False):
        mix = b["type"][b["type"] != ""].value_counts(normalize=True).mul(100).round(1)
        print(f"{tf:>4}: {len(b):>7,} bars  " + "  ".join(f"{t} {v}%" for t, v in mix.items()))
    print(f"wrote {a.outdir}/")


if __name__ == "__main__":
    main()
