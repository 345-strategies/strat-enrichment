"""Volume study engine (2026-10-24). Builds per-session 1-minute grids, volume baselines, in-force first-break
trades (S1), prior-day high/low breaks (S2) and per-session rows (S3-S5) with every fill and exit on 1-minute
bars, then writes trade tables for the analysis scripts.

    python volume/volstudy.py DATA.parquet NAME OUT_DIR --market futures|stocks|crypto

DATA.parquet: ts (bar start, UTC epoch seconds), o, h, l, c, v, and contract for futures (one contract per
session, the one you trade; see prep_futures.py). NAME labels the output files (s1_NAME.parquet ...) and is what
the analysis scripts look for (nq, es, spy, spyfwd, ...). Research only: nothing here is a trading rule."""
import os, sys, math, argparse
import numpy as np
import pandas as pd

COST = {"futures": 0.50, "stocks": 0.02, "crypto": None}   # round trip, in price units (points per contract, $ per share)
COST_PCT = {"crypto": 0.001}   # crypto: round trip as a share of the entry price (0.10%)
TZ = {"futures": "America/New_York", "stocks": "America/New_York", "crypto": "UTC"}
DEV_END = pd.Timestamp("2022-12-31").date()


def et(ts, tz="America/New_York"):
    return pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(tz)


# ---------------------------------------------------------------- loading
def load(path, market):
    """Read the input file: ts, o, h, l, c, v (+ contract for futures). Adds t (ET)."""
    d = pd.read_parquet(path)
    need = ["ts", "o", "h", "l", "c", "v"] + (["contract"] if market == "futures" else [])
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"{path}: missing columns {miss}")
    if "contract" not in d.columns:
        d["contract"] = "X"
    d = d[["ts", "o", "h", "l", "c", "v", "contract"]].copy()
    d["v"] = d["v"].astype(float)
    d = d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    d["t"] = et(d["ts"], TZ[market])
    return d


# ---------------------------------------------------------------- grids
def build(inst, d, market):
    fut = market == "futures"
    cry = market == "crypto"                         # 24h market: the session is the UTC day, all of it
    g0 = 0 if cry else (9 * 60 if fut else 9 * 60 + 30)   # grid start (minutes after midnight ET; UTC for crypto)
    END = 24 * 60 if cry else 16 * 60
    G = END - g0                                     # minutes in grid
    rth0 = 0 if cry else 9 * 60 + 30 - g0            # index of 09:30 (of 00:00 UTC for crypto)
    MINN = 1400 if cry else 380                      # minutes needed for a valid session
    mod = d.t.dt.hour * 60 + d.t.dt.minute
    day = d.t.dt.date
    # futures session key for D (18:00 start) and W
    if fut:
        sess = (d.t + pd.Timedelta(hours=6)).dt.date
    else:
        sess = day
    d = d.assign(mod=mod, day=day, sess=sess)
    # D and W opens on the traded contract
    d["dopen"] = d.groupby(["sess", "contract"])["o"].transform("first")
    wk = pd.to_datetime(d["sess"]).dt.isocalendar()
    d["wkey"] = wk["year"].astype(str) + "-" + wk["week"].astype(str)
    d["wopen"] = d.groupby(["wkey", "contract"])["o"].transform("first")
    if fut:
        on = d[(d["mod"] < 9 * 60 + 30) | (d["mod"] >= 18 * 60)]
        rthc = d[(d["mod"] >= 9 * 60 + 30) & (d["mod"] < 16 * 60)].groupby("day")["contract"].last()
        on = on[on["contract"].values == rthc.reindex(on["sess"]).values]
        onr = (on.groupby("sess")["h"].max() - on.groupby("sess")["l"].min())
        onv = on.groupby("sess")["v"].sum()
    else:
        onr = None; onv = None
    gm = d[(d["mod"] >= g0) & (d["mod"] < END)].copy()
    gm["mi"] = gm["mod"] - g0
    days = sorted(gm["day"].unique())
    S = len(days)
    di = {x: i for i, x in enumerate(days)}
    gm["si"] = gm["day"].map(di)
    A = {}
    for k in ("o", "h", "l", "c", "v", "dopen", "wopen"):
        arr = np.full((S, G), np.nan)
        arr[gm["si"].values, gm["mi"].values] = gm[k].values
        A[k] = arr
    con = gm.groupby("si")["contract"].agg(lambda x: x.iloc[-1]).reindex(range(S)).values
    ncon = gm.groupby("si")["contract"].nunique().reindex(range(S)).values
    nrth = np.isfinite(A["c"][:, rth0:]).sum(1)
    nvol = np.isfinite(A["v"][:, rth0:]).sum(1)
    valid = (nrth >= MINN) & (nvol >= MINN) & (ncon == 1)
    # fill holes: OHLC carried from the previous close, volume 0 (holes are minutes without trades)
    cff = pd.DataFrame(A["c"]).ffill(axis=1).values
    for k in ("o", "h", "l", "c"):
        m = np.isnan(A[k]); A[k][m] = cff[m]
    for k in ("dopen", "wopen"):
        A[k] = pd.DataFrame(A[k]).ffill(axis=1).values
    A["v"][np.isnan(A["v"])] = 0.0
    # roll exclusion (3 sessions before a contract switch)
    roll_x = np.zeros(S, bool)
    newcon = np.zeros(S, bool)
    if fut:
        for s in range(1, S):
            if con[s] != con[s - 1]:
                newcon[s] = True
                roll_x[max(0, s - 3):s] = True
    valid &= ~roll_x
    # leading NaNs in the grid (no prior close) -> invalid
    valid &= np.isfinite(A["c"]).all(1)
    # volume baseline: median per minute over prior 20 valid sessions
    med = np.full((S, G), np.nan)
    vidx = np.where(valid)[0]
    V = A["v"]
    for j, s in enumerate(vidx):
        if j >= 20:
            med[s] = np.median(V[vidx[j - 20:j]], axis=0)
    medb = np.full((S, G), np.nan)
    BODY = np.abs(A["c"] - A["o"])
    for j, s in enumerate(vidx):
        if j >= 20:
            medb[s] = np.median(BODY[vidx[j - 20:j]], axis=0)
    # ATR20: mean RTH range of prior 20 sessions with prices
    rr = np.nanmax(A["h"][:, rth0:], 1) - np.nanmin(A["l"][:, rth0:], 1)
    atr = pd.Series(rr).rolling(20, min_periods=20).mean().shift(1).values
    atr5 = pd.Series(rr).rolling(5, min_periods=5).mean().shift(1).values
    pdr = np.r_[np.nan, rr[:-1]]
    pdc = np.r_[np.nan, A["c"][:-1, G - 1]]
    onvol = onv.reindex(days).values if onv is not None else np.full(S, np.nan)
    if onr is not None:
        onrng = onr.reindex(days).values
    else:
        onrng = np.abs(A["o"][:, rth0] - pdc)
    pdh = np.r_[np.nan, np.nanmax(A["h"][:, rth0:], 1)[:-1]]
    pdl = np.r_[np.nan, np.nanmin(A["l"][:, rth0:], 1)[:-1]]
    usable = valid & np.isfinite(med).all(1) & np.isfinite(atr) & (np.nan_to_num(med).min(1) > 0)
    print(inst, "sessions", S, "valid", int(valid.sum()), "usable", int(usable.sum()), "roll-excluded", int(roll_x.sum()), flush=True)
    return dict(inst=inst, fut=fut, cost=COST[market], cost_pct=COST_PCT.get(market), g0=g0, G=G, rth0=rth0, days=np.array(days), A=A, med=med, medb=medb, atr=atr,
                atr5=atr5, pdr=pdr, onrng=onrng, onvol=onvol,
                pdh=pdh, pdl=pdl, usable=usable, newcon=newcon)


def bars(B, k):
    """Bar boundaries (start,end) in grid minutes. Futures grid starts 09:00 so 15m/30m/60m are clock aligned
    (60m 09:00-10:00 etc.); SPY grid starts 09:30 so its bars align to 09:30."""
    return [(x, x + k) for x in range(0, B["G"], k) if x + k <= B["G"]]


# ---------------------------------------------------------------- signals
def s1_trades(B, k):
    A, med, atr = B["A"], B["med"], B["atr"]
    O, H, L, C, V = A["o"], A["h"], A["l"], A["c"], A["v"]
    rth0, G = B["rth0"], B["G"]
    bb = bars(B, k)
    cost = B["cost"]
    out = []
    for s in np.where(B["usable"])[0]:
        a = atr[s]
        # 60m bar open for each minute
        o60 = np.full(G, np.nan)
        for x in range(0, G, 60):                    # includes a short last hour (stocks: 15:30-16:00 ET)
            o60[x:x + 60] = O[s, x]
        for j in range(1, len(bb)):
            c1s, c1e = bb[j - 1]
            c2s, c2e = bb[j]
            if c2s < rth0:
                continue
            c1h = H[s, c1s:c1e].max(); c1l = L[s, c1s:c1e].min()
            cc = C[s, c2s:c2e]
            up = cc > c1h; dn = cc < c1l
            iu = np.argmax(up) if up.any() else 99999
            idn = np.argmax(dn) if dn.any() else 99999
            if iu == 99999 and idn == 99999:
                continue
            side = 1 if iu < idn else -1
            tm = c2s + min(iu, idn)                     # trigger minute
            e = tm + 1
            if e >= G - 1:
                continue
            ent = O[s, e]
            if B["cost_pct"] is not None:
                cost = B["cost_pct"] * ent
            stop = c1l if side == 1 else c1h
            risk = (ent - stop) * side
            if not risk > 0:
                continue
            # C0 for type of C1
            if j >= 2:
                c0s, c0e = bb[j - 2]
                c0h = H[s, c0s:c0e].max(); c0l = L[s, c0s:c0e].min()
                bu, bd = c1h > c0h, c1l < c0l
                t1 = 3 if (bu and bd) else (1 if not (bu or bd) else (2 if bu else -2))
                bu0 = bd0 = None
                if j >= 3:
                    cms, cme = bb[j - 3]
                    cmh = H[s, cms:cme].max(); cml = L[s, cms:cme].min()
                    bu0, bd0 = c0h > cmh, c0l < cml
                    t0 = 3 if (bu0 and bd0) else (1 if not (bu0 or bd0) else (2 if bu0 else -2))
                else:
                    t0 = 0
            else:
                t1 = 0; t0 = 0
            # exits on 1m
            lo, hi, op = L[s, e:], H[s, e:], O[s, e:]
            if side == 1:
                hitst = lo <= stop
                tgt = ent + risk; hitt = hi >= tgt
            else:
                hitst = hi >= stop
                tgt = ent - risk; hitt = lo <= tgt
            if hitst.any():
                m = np.argmax(hitst)
                px = op[m] if (op[m] - stop) * side < 0 else stop
            else:
                m = None; px = C[s, G - 1]
            R = ((px - ent) * side - cost) / risk
            ms = np.argmax(hitst) if hitst.any() else 10**6
            mt = np.argmax(hitt) if hitt.any() else 10**6
            hit1 = np.nan if (ms == 10**6 and mt == 10**6) else (1.0 if mt < ms else 0.0)
            # features
            mv = med[s]
            rv_c1 = V[s, c1s:c1e].sum() / mv[c1s:c1e].sum()
            rv_trig = V[s, tm] / mv[tm]
            rv_c2 = V[s, c2s:tm + 1].sum() / mv[c2s:tm + 1].sum()
            rv_day = V[s, rth0:tm + 1].sum() / mv[rth0:tm + 1].sum()
            sg = np.sign(C[s, c1s:c1e] - O[s, c1s:c1e])
            vv = V[s, c1s:c1e]
            sv_c1 = side * (vv * sg).sum() / vv.sum() if vv.sum() > 0 else 0.0
            su_c1 = side * sg.mean()
            # round 4 management variants (1m path after entry)
            rv1 = V[s, e:] / med[s, e:]
            mv_sign = np.sign(C[s, e:] - O[s, e:])
            ms_i = ms if ms != 10**6 else len(lo)
            bx = np.abs(C[s, e:] - O[s, e:]) / np.maximum(B["medb"][s, e:], 1e-9)
            def climax_exit(want, mask=None):
                mk = (rv1[1:] >= 3.0) if mask is None else mask[1:]
                hit = np.where(mk & (mv_sign[1:] == want))[0]
                if len(hit) == 0:
                    return R
                m = hit[0] + 1
                if m >= ms_i:
                    return R
                if e + m + 1 >= G:
                    return ((C[s, G - 1] - ent) * side - cost) / risk
                return ((O[s, e + m + 1] - ent) * side - cost) / risk
            R_cxw = climax_exit(side); R_cxa = climax_exit(-side)
            R_pxw = climax_exit(side, bx >= 3.0); R_pxa = climax_exit(-side, bx >= 3.0)
            def stop_mult(k):
                st = ent - side * k * risk
                hs = (lo <= st) if side == 1 else (hi >= st)
                if hs.any():
                    mm = np.argmax(hs); pxx = op[mm] if (op[mm] - st) * side < 0 else st
                else:
                    pxx = C[s, G - 1]
                return ((pxx - ent) * side - cost) / (k * risk)
            R_w15 = stop_mult(1.5); R_t05 = stop_mult(0.5)
            mae_r = ((ent - lo.min()) if side == 1 else (hi.max() - ent)) / risk
            brk = c1h if side == 1 else c1l
            tb = int(((lo <= brk).any()) if side == 1 else ((hi >= brk).any()))
            R1R = (1 - cost / risk) if (mt < ms) else R
            px_t = C[s, tm]
            ao60 = np.sign(px_t - o60[tm]) if px_t != o60[tm] else 1
            aod = 1 if px_t >= A["dopen"][s, tm] else -1
            aow = 1 if px_t >= A["wopen"][s, tm] else -1
            ao60 = 1 if px_t >= o60[tm] else -1
            tfc = int(ao60 == side and aod == side and aow == side)
            sd_h = H[s, rth0:tm + 1].max(); sd_l = L[s, rth0:tm + 1].min()
            out.append(dict(si=s, day=B["days"][s], tf=k, side=side, tmin=tm, mins_left=G - 1 - e, ent=ent, risk=risk,
                            risk_atr=risk / a, R=R, hit1=hit1, stopped=int(hitst.any()),
                            rng_rest=(H[s, e:].max() - L[s, e:].min()) / a,
                            c1rng=(c1h - c1l) / a, sdrng=(sd_h - sd_l) / a,
                            t1=t1 * (side if abs(t1) == 2 else 1), t0=t0, tfc=tfc, ao60=ao60 * side, aod=aod * side, aow=aow * side,
                            rv_c1=rv_c1, rv_trig=rv_trig, rv_c2=rv_c2, rv_day=rv_day, sv_c1=sv_c1,
                            su_c1=su_c1, R1R=R1R, R_cxw=R_cxw, R_cxa=R_cxa, R_pxw=R_pxw, R_pxa=R_pxa, R_w15=R_w15, R_t05=R_t05, mae_r=mae_r, tb=tb, a5=B["atr5"][s] / a, pdr=B["pdr"][s] / a, onr=B["onrng"][s] / a))
    return pd.DataFrame(out)


def s2_breaks(B):
    A, med, atr = B["A"], B["med"], B["atr"]
    O, H, L, C, V = A["o"], A["h"], A["l"], A["c"], A["v"]
    rth0, G = B["rth0"], B["G"]
    out = []
    for s in np.where(B["usable"])[0]:
        if s == 0 or B["newcon"][s] or not np.isfinite(B["pdh"][s]):
            continue
        a = atr[s]
        cc = C[s, rth0:]
        for side, lvl in ((1, B["pdh"][s]), (-1, B["pdl"][s])):
            br = (cc > lvl) if side == 1 else (cc < lvl)
            if not br.any():
                continue
            tm = rth0 + int(np.argmax(br))
            if tm >= G - 2:
                continue
            # skip if the session opened beyond the level (gap) -- still a break; keep, flagged
            gap = int(((O[s, rth0] - lvl) * side) > 0)
            mv = med[s]
            tb2 = int((L[s, tm + 1:] <= lvl).any() if side == 1 else (H[s, tm + 1:] >= lvl).any())
            out.append(dict(si=s, day=B["days"][s], side=side, tmin=tm, gap=gap, tb2=tb2,
                            expand=int((C[s, G - 1] - lvl) * side > 0),
                            move=(C[s, G - 1] - C[s, tm]) * side / a,
                            rv_trig=V[s, tm] / mv[tm],
                            rv_day=V[s, rth0:tm + 1].sum() / mv[rth0:tm + 1].sum()))
    return pd.DataFrame(out)


def s3_sessions(B):
    """Round 2 E4: one row per usable session, decision at 10:30 ET."""
    A, med, atr = B["A"], B["med"], B["atr"]
    H, L, V = A["h"], A["l"], A["v"]
    rth0, G = B["rth0"], B["G"]
    k = rth0 + 60
    k15 = rth0 + 15
    out = []
    totv = V[:, rth0:].sum(1); totm = med[:, rth0:].sum(1)
    ov = pd.Series(B["onvol"])
    ov_med = ov.rolling(20, min_periods=20).median().shift(1).values
    for s in np.where(B["usable"])[0]:
        a = atr[s]
        k30 = rth0 + 30
        ext = dict(dayr=(H[s, rth0:].max() - L[s, rth0:].min()) / a, atr=a,
                   fh30=(H[s, rth0:k30].max() - L[s, rth0:k30].min()) / a,
                   rest30=(H[s, k30:].max() - L[s, k30:].min()) / a,
                   rv_30=V[s, rth0:k30].sum() / med[s, rth0:k30].sum(),
                   rest15=(H[s, k15:].max() - L[s, k15:].min()) / a,
                   fh15=(H[s, rth0:k15].max() - L[s, rth0:k15].min()) / a,
                   rv_15=V[s, rth0:k15].sum() / med[s, rth0:k15].sum(),
                   rv_pd=(totv[s - 1] / totm[s - 1]) if s > 0 and np.isfinite(totm[s - 1]) and totm[s - 1] > 0 else np.nan,
                   rv_on=(B["onvol"][s] / ov_med[s]) if np.isfinite(ov_med[s]) and ov_med[s] > 0 else np.nan)
        out.append(dict(si=s, day=B["days"][s], fh=(H[s, rth0:k].max() - L[s, rth0:k].min()) / a,
                        rest=(H[s, k:].max() - L[s, k:].min()) / a,
                        rv_fh=V[s, rth0:k].sum() / med[s, rth0:k].sum(),
                        a5=B["atr5"][s] / a, pdr=B["pdr"][s] / a, onr=B["onrng"][s] / a, **ext))
    return pd.DataFrame(out)


def s4_daily(B):
    """Round 5 literature tests: one row per usable session (Gao et al. and CGW inputs)."""
    A, med = B["A"], B["med"]
    C, V = A["c"], A["v"]
    rth0, G = B["rth0"], B["G"]
    days = B["days"]
    S = len(days)
    con = B.get("con")
    totv = V[:, rth0:].sum(1)
    lv = np.log(np.where(totv > 0, totv, np.nan))
    lvm = pd.Series(lv).rolling(250, min_periods=120).mean().shift(1).values
    pc = np.r_[np.nan, C[:-1, G - 1]]
    out = []
    for s in np.where(B["usable"])[0]:
        if s == 0 or B["newcon"][s]:
            continue
        i10 = rth0 + 30; i1529 = G - 31; i1459 = G - 61
        out.append(dict(si=s, day=days[s], r1=C[s, i10 - 1] / pc[s] - 1, r12=C[s, i1529] / C[s, i1459] - 1,
                        r13=C[s, G - 1] / C[s, i1529] - 1, rv_30=V[s, rth0:i10].sum() / med[s, rth0:i10].sum(),
                        rd=C[s, G - 1] / pc[s] - 1, vdt=lv[s] - lvm[s], nextok=int(s + 1 < S and B["usable"][s + 1] and not B["newcon"][s + 1])))
    return pd.DataFrame(out)


def s5_bars30(B):
    """Round 5 LIT2b: consecutive 30m RTH bar returns with RVOL, within a session."""
    A, med = B["A"], B["med"]
    O, C, V = A["o"], A["c"], A["v"]
    rth0, G = B["rth0"], B["G"]
    out = []
    starts = list(range(rth0, G, 30))
    for s in np.where(B["usable"])[0]:
        prev = None
        for x in starts:
            y = x + 30
            if y > G:
                break
            r = C[s, y - 1] / (C[s, x - 1] if x > 0 else O[s, x]) - 1
            rv = V[s, x:y].sum() / med[s, x:y].sum()
            if prev is not None:
                out.append(dict(si=s, day=B["days"][s], r0=prev[0], rv0=prev[1], r1=r))
            prev = (r, rv)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Volume study engine: build trade tables from 1-minute bars")
    ap.add_argument("data"); ap.add_argument("name"); ap.add_argument("out")
    ap.add_argument("--market", choices=["futures", "stocks", "crypto"], required=True)
    ap.add_argument("--cost-pct", type=float, help="round-trip cost as a share of entry price (e.g. 0.0002 = 2 bp), "
                                                   "in place of the market's default")
    a = ap.parse_args()
    inst, od = a.name, a.out
    os.makedirs(od, exist_ok=True)
    d = load(a.data, a.market)
    B = build(inst, d, a.market)
    if a.cost_pct is not None:
        B["cost"], B["cost_pct"] = 0.0, a.cost_pct
    ts = []
    for k in (15, 30, 60):
        t = s1_trades(B, k)
        print(inst, k, "trades", len(t), flush=True)
        ts.append(t)
    t = pd.concat(ts, ignore_index=True)
    t["inst"] = inst
    t.to_parquet(f"{od}/s1_{inst}.parquet", index=False)
    s2 = s2_breaks(B); s2["inst"] = inst
    s2.to_parquet(f"{od}/s2_{inst}.parquet", index=False)
    print(inst, "s2 breaks", len(s2), flush=True)
    s4_daily(B).assign(inst=inst).to_parquet(f"{od}/s4_{inst}.parquet", index=False)
    s5_bars30(B).assign(inst=inst).to_parquet(f"{od}/s5_{inst}.parquet", index=False)
    s3 = s3_sessions(B); s3["inst"] = inst
    s3.to_parquet(f"{od}/s3_{inst}.parquet", index=False)


if __name__ == "__main__":
    main()
