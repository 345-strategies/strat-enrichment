"""Strat Stats: a descriptive atlas of first breaks, from 1-minute bars.

Builds session-anchored bars with strat_enrichment (bar types from the public
grammar), then measures every first break of a closed setup bar on the next
bar, sequenced on 1-minute data:
  long  = first trade through the setup high, short = through the setup low
  stop  = the other side of the setup bar, one tick beyond
  R     = |entry - stop|; outcomes over the trigger bar plus two more bars
  the entry minute is excluded from stop/target checks (order unknowable);
  a minute touching stop and target counts as the stop.
Every cell is compared with all first breaks in the same timeframe and
direction. Wilson 95% intervals, z-tests, Benjamini-Hochberg q per table.

Usage: python stats/atlas.py <futures|stocks> <minutes.parquet> <out.json> [--name ES] [--tick 0.25] [--timeframes ...]
minutes.parquet columns: ts (UTC epoch seconds, bar start), o, h, l, c; futures add contract
(raw contract prices; each roll is back-adjusted).
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from thestrat_grammar import Candle

from strat_enrichment import back_adjust, build_bars, classify_bars, sessionize
from strat_enrichment.bars import tf_minutes

ET = ZoneInfo("America/New_York")
CFG = {
    "futures": dict(tick=0.25, tfs="15m,30m,1h,4h,D,W"),
    "stocks": dict(tick=0.01, tfs="15m,30m,1h,D,W"),
}
SPAN = {"D": 1440, "W": 10080}
MIN_N = 100
STUB = 0.6


# ------------------------------------------------------------------ bars
def bars_for(m, tf):
    """Bars for one timeframe with the atlas's labels. Types come from the grammar;
    stubs (under 60% of the usual minutes for that slot, e.g. holiday sessions) are
    flagged and kept out of the trades. Also returns each minute's bar index."""
    b, bar_index = build_bars(m, tf)
    b = pd.concat([b, classify_bars(b)], axis=1)
    n = tf_minutes(tf)
    b["slot"] = m.since.to_numpy()[b.i0.to_numpy()] // n if n else 0
    b["sd"] = [d.toordinal() for d in pd.to_datetime(b.session).dt.date]
    b["n"] = b.minutes
    b["stub"] = b.n < STUB * b.groupby("slot").n.transform("median")
    note = b.type                                   # grammar notation: 1u, 2d, F2u, 3d, ...
    b["failed"] = note.str.startswith("F")
    b["disp"] = np.where(note.str.startswith("1"), "1", note.str.upper())          # 1, 2U, F2U, 2D, F2D, 3U, 3D
    b["type"] = np.where(note.str.startswith("3"), "3", pd.Series(b.disp, index=b.index).str.removeprefix("F"))
    b["green"] = b.above_open                       # sign channel: a close at the open is not above
    b["eqh"] = b.h == b.h.shift(1)
    b["eql"] = b.l == b.l.shift(1)
    return b, bar_index


# ----------------------------------------------------------- the trades
def first_breaks(b, m, tick, htf_ctx):
    """One trade per setup bar whose next bar breaks it. Sequenced on 1m."""
    H, L, O, C = m.h.values, m.l.values, m.o.values, m.c.values
    out = []
    n = len(b)
    amb = 0
    for i in range(2, n - 3):
        S, T = b.iloc[i], b.iloc[i + 1]
        if S.stub or T.stub or b.iloc[i - 1].stub or b.iloc[i + 2].stub or b.iloc[i + 3].stub:
            continue
        a0, a1 = int(T.i0), int(T.i1)
        hu = np.flatnonzero(H[a0:a1 + 1] > S.h)
        ld = np.flatnonzero(L[a0:a1 + 1] < S.l)
        if not len(hu) and not len(ld):
            continue
        ju = hu[0] if len(hu) else 10 ** 9
        jd = ld[0] if len(ld) else 10 ** 9
        if ju == jd:
            amb += 1
            continue
        d = 1 if ju < jd else -1
        e_i = a0 + min(ju, jd)
        if d > 0:
            entry = max(S.h, O[e_i]) if O[e_i] > S.h else S.h
            stop = S.l - tick
        else:
            entry = min(S.l, O[e_i]) if O[e_i] < S.l else S.l
            stop = S.h + tick
        R = abs(entry - stop)
        if R <= 0:
            continue
        end = int(b.iloc[i + 3].i1)
        hh, ll = H[e_i + 1:end + 1], L[e_i + 1:end + 1]
        if d > 0:
            st = np.flatnonzero(ll <= stop)
            t1 = np.flatnonzero(hh >= entry + R)
            t2 = np.flatnonzero(hh >= entry + 2 * R)
            fav = (hh.max() - entry) / R if len(hh) else 0.0
            adv = (entry - ll.min()) / R if len(ll) else 0.0
        else:
            st = np.flatnonzero(hh >= stop)
            t1 = np.flatnonzero(ll <= entry - R)
            t2 = np.flatnonzero(ll <= entry - 2 * R)
            fav = (entry - ll.min()) / R if len(ll) else 0.0
            adv = (hh.max() - entry) / R if len(hh) else 0.0
        s0 = st[0] if len(st) else 10 ** 9

        def first(tt):
            x = tt[0] if len(tt) else 10 ** 9
            if x == 10 ** 9 and s0 == 10 ** 9:
                return None
            return 1 if x < s0 else 0   # same minute -> stop
        # MFE before the stop
        if s0 < 10 ** 9:
            seg_h, seg_l = hh[:s0 + 1], ll[:s0 + 1]
            fav_s = ((seg_h.max() - entry) if d > 0 else (entry - seg_l.min())) / R
        else:
            fav_s = fav
        cb = (T.c > entry) if d > 0 else (T.c < entry)
        f2 = (T.c <= S.h) if d > 0 else (T.c >= S.l)    # trigger bar closes back inside: a failed 2
        P = b.iloc[i - 1]
        row = dict(i=i, ts=int(S.ts), sd=int(S.sd), year=datetime.fromordinal(int(S.sd)).year, slot=int(S.slot),
                   dir=d, c2=P.disp, c1=S.disp, c2t=P.type, c1t=S.type, c1green=bool(S.green),
                   R=R, Rpct=R / entry * 100, gap=bool(O[e_i] != (S.h if d > 0 else S.l) and
                                                    ((O[e_i] > S.h) if d > 0 else (O[e_i] < S.l))),
                   p1=first(t1), p2=first(t2), stop=s0 < 10 ** 9, cb=bool(cb), f2=bool(f2),
                   mfe=fav_s, mae=min(adv, 1.0 + tick / R) if s0 < 10 ** 9 else adv,
                   rng_atr=None)
        row.update(htf_ctx(S, d))
        out.append(row)
    return pd.DataFrame(out), amb


# ----------------------------------------------------------- statistics
def wilson(k, n):
    if n == 0:
        return None, None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h) * 100, (c + h) * 100


def ztest(k, n, p0):
    if n == 0 or p0 <= 0 or p0 >= 1:
        return None
    z = (k / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return math.erfc(abs(z) / math.sqrt(2))


def bh(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]
    q = [None] * len(ps)
    if not idx:
        return q
    srt = sorted(idx, key=lambda i: ps[i])
    m = len(srt)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = srt[rank - 1]
        prev = min(prev, ps[i] * m / rank)
        q[i] = prev
    return q


def cell(df, base):
    """Summary of a group of trades against its baseline (dict of base rates)."""
    n = len(df)
    res = df.p1.notna()
    k1 = int((df.p1 == 1).sum())
    n1 = int(res.sum())
    res2 = df.p2.notna()
    k2 = int((df.p2 == 1).sum())
    n2 = int(res2.sum())
    kcb = int(df.cb.sum())
    kf2 = int(df.f2.sum())
    out = dict(n=n, n1=n1,
               p1=k1 / n1 * 100 if n1 else None, p1_ci=wilson(k1, n1),
               p2=k2 / n2 * 100 if n2 else None,
               cb=kcb / n * 100 if n else None, f2=kf2 / n * 100 if n else None,
               res=n1 / n * 100 if n else None,
               mfe=float(df.mfe.median()) if n else None, mae=float(df.mae.median()) if n else None,
               rpct=float(df.Rpct.median()) if n else None)
    if base is not None and n >= MIN_N:
        out["d1"] = out["p1"] - base["p1"] if n1 else None
        out["dcb"] = out["cb"] - base["cb"]
        out["df2"] = out["f2"] - base["f2"]
        out["pz1"] = ztest(k1, n1, base["p1"] / 100) if n1 else None
        out["pzcb"] = ztest(kcb, n, base["cb"] / 100)
        out["pzf2"] = ztest(kf2, n, base["f2"] / 100)
    return out


def r(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    if isinstance(x, tuple):
        return [r(v, nd) for v in x]
    return round(float(x), nd)


def tidy(d):
    return {k: (v if isinstance(v, str) or isinstance(v, (bool, np.bool_)) else r(v, 4) if k.startswith(("p", "q")) and k not in ("p1", "p2") else r(v)) for k, v in d.items()}


def grouped(trades, keys, bases, extra=None):
    """Rows for each group of trades by keys (always within tf and dir), with BH q."""
    rows = []
    for (tf, d), tdf in trades.groupby(["tf", "dir"]):
        base = bases[(tf, d)]
        for k, g in tdf.groupby(keys, dropna=False):
            k = k if isinstance(k, tuple) else (k,)
            c = cell(g, base)
            c.update(tf=tf, dir="long" if d > 0 else "short", **dict(zip(keys, k)))
            rows.append(c)
    for tf in {x["tf"] for x in rows}:
        sub = [x for x in rows if x["tf"] == tf]
        for metric in ("pz1", "pzcb", "pzf2"):
            qs = bh([x.get(metric) for x in sub])
            for x, q in zip(sub, qs):
                x["q" + metric[2:]] = q
    return [tidy(x) for x in rows]


# ------------------------------------------------------------------ main
def span(tf):
    return tf_minutes(tf) or SPAN[tf]


def main():
    p = argparse.ArgumentParser(description="Strat Stats atlas from 1-minute bars.")
    p.add_argument("market", choices=sorted(CFG), help="futures: sessions open 18:00 ET; stocks: 09:30-16:00 ET")
    p.add_argument("src", type=Path, help="1-minute parquet: ts, o, h, l, c (futures: raw prices plus contract)")
    p.add_argument("dst", type=Path, help="atlas JSON for build_page.py")
    p.add_argument("--name", help="market label on the page (default: input file name)")
    p.add_argument("--tick", type=float, help="tick size for the stop (default: 0.25 futures, 0.01 stocks)")
    p.add_argument("--timeframes", help="comma-separated, D and W last (default: 15m,30m,1h,4h,D,W futures; 15m,30m,1h,D,W stocks)")
    a = p.parse_args()
    cfg = CFG[a.market]
    tick = a.tick or cfg["tick"]
    tfs = sorted((a.timeframes or cfg["tfs"]).split(","), key=span)
    trade_tfs = [tf for tf in tfs if tf != "W"]

    m = pd.read_parquet(a.src).sort_values("ts").reset_index(drop=True)
    if "contract" in m.columns:
        m, rolls = back_adjust(m)
        print(f"{len(rolls)} contract rolls back-adjusted", flush=True)
    m = sessionize(m, a.market)
    bars, minute_bar, run_h, run_l = {}, {}, {}, {}
    for tf in tfs:
        bars[tf], minute_bar[tf] = bars_for(m, tf)
        run_h[tf] = m.h.groupby(minute_bar[tf]).cummax().to_numpy()
        run_l[tf] = m.l.groupby(minute_bar[tf]).cummin().to_numpy()
        print(f"{tf}: {len(bars[tf])} bars, {int(bars[tf].stub.sum())} stubs", flush=True)

    # higher-timeframe context at a setup bar's close: sign (above or below open) of each
    # forming higher bar, and whether the next timeframe up is in force (beyond its last
    # completed bar's extreme)
    def ctx_for(tf):
        higher = [x for x in tfs if span(x) > span(tf)]
        nxt = higher[0] if higher else None

        def f(S, d):
            last = int(S.i1)
            close = S.c
            agree = []
            for h in higher:
                hb = bars[h].iloc[minute_bar[h][last]]
                # the forming higher bar so far; if the setup closes the higher bar too, it is complete
                up = Candle(hb.o, run_h[h][last], run_l[h][last], close).above_open
                agree.append(1 if up == (d > 0) else 0)
            share = sum(agree) / len(agree) if agree else None
            ftc = "aligned" if share == 1 else ("opposed" if share == 0 else "mixed")
            inf = None
            if nxt:
                j = minute_bar[nxt][last]
                if j >= 1:
                    pb = bars[nxt].iloc[j - 1]
                    hi_sofar, lo_sofar = run_h[nxt][last], run_l[nxt][last]
                    up = close > pb.h
                    dn = close < pb.l
                    # in force = price beyond the higher bar's prior extreme at the setup close
                    inf = "with" if (up and d > 0) or (dn and d < 0) else ("against" if (up or dn) else "none")
                    # failing = broke the prior extreme but price is back inside at the setup close
                    if inf == "none" and (hi_sofar > pb.h or lo_sofar < pb.l):
                        inf = "failing"
            return dict(ftc=ftc, ftc_share=share, nxt_inforce=inf)
        return f

    T = []
    meta = []
    for tf in tfs:
        if tf not in trade_tfs:
            continue
        b = bars[tf]
        tr, amb = first_breaks(b, m, tick, ctx_for(tf))
        # setup-bar range against the 14-bar ATR of that timeframe (known at the setup close)
        tr_ = pd.concat([b.h - b.l, (b.h - b.c.shift()).abs(), (b.l - b.c.shift()).abs()], axis=1).max(axis=1)
        atr = tr_.rolling(14).mean().shift(1)
        tr["rng_atr"] = [(b.h.iloc[i] - b.l.iloc[i]) / atr.iloc[i] if atr.iloc[i] > 0 else None for i in tr.i]
        tr["tf"] = tf
        T.append(tr)
        meta.append(dict(tf=tf, bars=len(b), stubs=int(b.stub.sum()), events=len(tr), ambiguous=amb,
                         tie_high=r(b.eqh.mean() * 100), tie_low=r(b.eql.mean() * 100)))
        print(f"{tf}: {len(tr)} first breaks, {amb} same-minute ambiguous", flush=True)
    trades = pd.concat(T, ignore_index=True)

    bases = {}
    base_rows = []
    for (tf, d), g in trades.groupby(["tf", "dir"]):
        c = cell(g, None)
        bases[(tf, d)] = c
        base_rows.append(tidy(dict(c, tf=tf, dir="long" if d > 0 else "short")))

    # bar-level tables
    freq, trans, f2tab, years_bar = [], [], [], []
    for tf in tfs:
        b = bars[tf]
        kb = b[(b.type != "") & ~b.stub]
        vc = kb.type.value_counts(normalize=True) * 100
        fr = kb.groupby("type").failed.mean() * 100
        for t in ("1", "2U", "2D", "3"):
            freq.append(dict(tf=tf, type=t, pct=r(vc.get(t, 0)), failed_pct=r(fr.get(t)) if t in ("2U", "2D") else None))
        nxt = kb.type.shift(-1)
        dnext = kb.disp
        uncond = kb.type.value_counts(normalize=True)
        for frm, g in kb.assign(nx=nxt).dropna(subset=["nx"]).groupby("disp"):
            if len(g) < MIN_N:
                continue
            vc2 = g.nx.value_counts()
            n = len(g)
            row = dict(tf=tf, frm=frm, n=n)
            for t in ("1", "2U", "2D", "3"):
                k = int(vc2.get(t, 0))
                row[t] = r(k / n * 100)
                row["d" + t] = r((k / n - uncond.get(t, 0)) * 100)
                row["p" + t] = ztest(k, n, uncond.get(t, 0))
            trans.append(row)
        # failed 2s: of each 2U/2D, share that closed back inside; then what the next bar did
        for t in ("2U", "2D"):
            g = kb[kb.type == t].copy()
            g["nx"] = nxt[g.index]
            for fl, gg in g.groupby("failed"):
                gg = gg.dropna(subset=["nx"])
                if len(gg) < 30:
                    continue
                vc3 = gg.nx.value_counts(normalize=True) * 100
                f2tab.append(dict(tf=tf, type=t, failed=bool(fl), n=len(gg),
                                  **{"next_" + x: r(vc3.get(x, 0)) for x in ("1", "2U", "2D", "3")}))
    # BH within each tf for transitions
    for tf in {x["tf"] for x in trans}:
        sub = [x for x in trans if x["tf"] == tf]
        ps = [x[k] for x in sub for k in ("p1", "p2U", "p2D", "p3")]
        qs = bh(ps)
        it = iter(qs)
        for x in sub:
            for t in ("1", "2U", "2D", "3"):
                x["q" + t] = r(next(it), 4)
                x["p" + t] = r(x["p" + t], 4)

    # trade tables
    setups = grouped(trades, ["c2", "c1"], bases)
    setups_type = grouped(trades, ["c2t", "c1t"], bases)
    ftc = grouped(trades, ["ftc"], bases)
    inforce = grouped(trades, ["nxt_inforce"], bases)
    trades["rng_bin"] = pd.cut(trades.rng_atr, [0, 0.5, 0.75, 1, 1.5, 2, 99], labels=["<0.5", "0.5-0.75", "0.75-1", "1-1.5", "1.5-2", ">2"])
    rng = grouped(trades, ["rng_bin"], bases)
    trades["colour"] = np.where(trades.c1green == (trades.dir > 0), "with trade", "against trade")
    colour = grouped(trades, ["colour"], bases)
    trades["c1failed"] = trades.c1.str.startswith("F")
    c1f = grouped(trades, ["c1failed"], bases)
    year_base = grouped(trades, ["year"], bases)
    trades["period"] = np.where(trades.year <= 2022, "dev (to 2022)", "holdout (2023+)")
    period_setup = grouped(trades, ["period", "c2", "c1"], bases)
    quant = []
    for (tf, d), g in trades.groupby(["tf", "dir"]):
        for metric in ("mfe", "mae"):
            q = g[metric].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
            quant.append(dict(tf=tf, dir="long" if d > 0 else "short", metric=metric, n=len(g),
                              mean=r(g[metric].mean()), q=[r(v) for v in q]))
    first_ts, last_ts = int(m.ts.iloc[0]), int(m.ts.iloc[-1])
    info = dict(market=a.name or a.src.stem.upper(), session=a.market, start=datetime.fromtimestamp(first_ts, ET).date().isoformat(),
                end=datetime.fromtimestamp(last_ts, ET).date().isoformat(), minutes=len(m),
                tfs=tfs, trade_tfs=trade_tfs, min_n=MIN_N)
    out = dict(info=info, meta=meta, base=base_rows, freq=freq, trans=trans, failed=f2tab, setups=setups,
               setups_type=setups_type, ftc=ftc, inforce=inforce, rng=rng, colour=colour, c1failed=c1f,
               year=year_base, period=period_setup, quant=quant)
    json.dump(out, open(a.dst, "w"), default=lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else (bool(x) if isinstance(x, np.bool_) else (int(x) if isinstance(x, np.integer) else float(x))))
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
