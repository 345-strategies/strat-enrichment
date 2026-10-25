"""Volume study round 4 (PREREG_ROUND4.md): C confirmation of lead L1 on new data, M management on 1m,
Z day-range sizing. Usage: python analyze4.py <work dir> <out.md>"""
import os, sys
import numpy as np
import pandas as pd
from analyze import ols_cl, DEV_END

LF = 0.05
CONF = ["iwmfwd", "aaplfwd", "nvdafwd", "tslafwd", "eslate"]


def lg(x):
    return np.log(np.maximum(np.asarray(x, float), LF))


def terc(x):
    lo, hi = np.nanquantile(x, [1 / 3, 2 / 3])
    return np.where(x >= hi, 1, np.where(x <= lo, -1, 0))


def cdiff(y, g, cl):
    m = (g != 0) & np.isfinite(y)
    if (g[m] == 1).sum() < 5 or (g[m] == -1).sum() < 5:
        return np.nan, np.nan, 0, 0
    X = np.c_[np.ones(m.sum()), (g[m] == 1).astype(float)]
    b, se = ols_cl(X, y[m], cl[m])
    return b[1], b[1] / se[1], int((g[m] == 1).sum()), int((g[m] == -1).sum())


def mean_t(y, cl):
    m = np.isfinite(y)
    b, se = ols_cl(np.ones((m.sum(), 1)), y[m], cl[m])
    return b[0], b[0] / se[0] if se[0] > 0 else np.nan


def rd(p):
    return pd.read_parquet(p) if os.path.exists(p) else None


def main():
    w, outp = sys.argv[1], sys.argv[2]
    L = ["# Volume study round 4 (PREREG_ROUND4.md)\n"]
    # ---------------- C
    L.append("## C. Confirmation of L1 on new data: high vs low trigger-minute RVOL, P(close beyond the level)\n")
    L.append("| instrument | window | breaks | top minus bottom (pts) | t | top / bottom n | throwback-free move to close, top minus bottom (ATR) |\n|---|---|---|---|---|---|---|")
    pool = []
    for i in CONF:
        s2 = rd(f"{w}/s2_{i}.parquet")
        if s2 is None or len(s2) == 0:
            L.append(f"| {i} | missing | | | | | |"); continue
        s2 = s2.copy(); s2["day"] = pd.to_datetime(s2.day)
        if i == "eslate":
            s2 = s2[s2.day >= pd.Timestamp("2026-05-15")]
        s2 = s2.reset_index(drop=True)
        g = terc(s2.rv_trig.values)
        cl = s2.day.dt.strftime("%Y%m%d").values
        d, t, na, nb = cdiff(s2.expand.values.astype(float), g, cl)
        dm, tm, _, _ = cdiff(s2.move.values, g, cl)
        L.append(f"| {i} | {s2.day.min().date()}..{s2.day.max().date()} | {len(s2)} | {d*100:+.1f} | {t:+.2f} | {na} / {nb} | {dm:+.3f} (t {tm:+.2f}) |")
        pool.append(pd.DataFrame(dict(y=s2.expand.values.astype(float), mv=s2.move.values, g=g, cl=cl)))
    if pool:
        P = pd.concat(pool, ignore_index=True)
        d, t, na, nb = cdiff(P.y.values, P.g.values, P.cl.values)
        dm, tm, _, _ = cdiff(P.mv.values, P.g.values, P.cl.values)
        L.append(f"| **pooled** | | {len(P)} | **{d*100:+.1f}** | **{t:+.2f}** | {na} / {nb} | {dm:+.3f} (t {tm:+.2f}) |")
        L.append(f"\nConfirmation rule: pooled difference negative with t <= -1.65: **{'CONFIRMED' if (t <= -1.65) else 'NOT CONFIRMED'}**.\n")
    # ---------------- M
    L.append("## M. Management on 1m (S1 in-force breaks), mean per-trade difference in net R, session-clustered\n")
    L.append("| inst | part | n | M1 vol-climax exit minus price-climax exit | M1b vol-climax exit minus hold | M2 against-climax: vol minus price | M2b against-climax vol minus hold | M3 1.5x stop gain, high minus low rv_trig | M3b 0.5x stop gain, high minus low | M4 hold-vs-+1R gain, high minus low rv_day |\n|---|---|---|---|---|---|---|---|---|---|")
    MR = {}
    for i in ["nq", "es", "spy", "fwd"]:
        if i == "fwd":
            parts = [rd(f"{w}/s1_{x}.parquet") for x in ("spyfwd", "qqqfwd")]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            t = pd.concat(parts, ignore_index=True)
        else:
            t = rd(f"{w}/s1_{i}.parquet")
            if t is None:
                continue
        t = t[t.risk_atr >= 0.05].copy().reset_index(drop=True)
        t["day"] = pd.to_datetime(t.day)
        t["cl"] = t.inst.astype(str).values + "_" + t.day.dt.strftime("%Y%m%d").values if "inst" in t else t.day.dt.strftime("%Y%m%d").values
        if i == "fwd":
            splits = [("forward", np.ones(len(t), bool))]
        else:
            dv = (t.day <= DEV_END).values
            splits = [("dev", dv), ("holdout", ~dv)]
        for nm, m in splits:
            x = t[m].reset_index(drop=True)
            cl = x.day.dt.strftime("%Y%m%d").values
            cells = []
            for col in [x.R_cxw - x.R_pxw, x.R_cxw - x.R, x.R_cxa - x.R_pxa, x.R_cxa - x.R]:
                b, tt = mean_t(col.values, cl); cells.append((b, tt))
            # terciles within tf (dev cut points for nq/es/spy; within-sample for forward)
            def gl(f):
                g = np.zeros(len(x), int)
                for k, ins in [(k, ins) for k in (15, 30, 60) for ins in x.inst.unique()]:
                    mk = ((x.tf == k) & (x.inst == ins)).values
                    if not mk.any():
                        continue
                    ref = t[(t.tf == k) & (t.inst == ins) & ((t.day <= DEV_END) if i != "fwd" else True)][f].values
                    lo, hi = np.nanquantile(ref, [1 / 3, 2 / 3])
                    v = x.loc[mk, f].values
                    g[mk] = np.where(v >= hi, 1, np.where(v <= lo, -1, 0))
                return g
            gt, gd = gl("rv_trig"), gl("rv_day")
            for col, g in [((x.R_w15 - x.R).values, gt), ((x.R_t05 - x.R).values, gt), ((x.R - x.R1R).values, gd)]:
                d, tt, _, _ = cdiff(col, g, cl); cells.append((d, tt))
            MR[(i, nm)] = cells
            L.append(f"| {i} | {nm} | {len(x):,} | " + " | ".join(f"{b:+.4f} ({tt:+.1f})" for b, tt in cells) + " |")
    L.append("\nPass rule per M test: same sign dev and holdout on ES and SPY, holdout |t| >= 2 on both, same sign forward.\n")
    names = ["M1", "M1b", "M2", "M2b", "M3", "M3b", "M4"]
    L.append("| test | ES dev/holdout | SPY dev/holdout | forward | PASS |\n|---|---|---|---|---|")
    for j, n in enumerate(names):
        try:
            ed, eh = MR[("es", "dev")][j], MR[("es", "holdout")][j]
            sd, sh = MR[("spy", "dev")][j], MR[("spy", "holdout")][j]
            fw = MR.get(("fwd", "forward"), [(np.nan, np.nan)] * 7)[j]
        except KeyError:
            continue
        sg = np.sign(eh[0])
        ok = (np.sign(ed[0]) == sg and np.sign(sd[0]) == sg and np.sign(sh[0]) == sg and abs(eh[1]) >= 2 and abs(sh[1]) >= 2 and np.sign(fw[0]) == sg)
        L.append(f"| {n} | {ed[0]:+.4f} / {eh[0]:+.4f} (t {eh[1]:+.1f}) | {sd[0]:+.4f} / {sh[0]:+.4f} (t {sh[1]:+.1f}) | {fw[0]:+.4f} (t {fw[1]:+.1f}) | {'**PASS**' if ok else 'fail'} |")
    # ---------------- Z
    L.append("\n## Z. Sizing: does volume forecast the day's range? (holdout MSE gain of log range vs price-only)\n")
    L.append("| inst | test | sessions fit / scored | MSE gain from volume | volume coefs dev (t) / holdout (t) | PASS |\n|---|---|---|---|---|---|")
    S3 = {i: rd(f"{w}/s3_{i}.parquet") for i in ["nq", "es", "spy", "spyfwd"]}
    for i, s in S3.items():
        if s is None:
            continue
        s = s.copy(); s["day"] = pd.to_datetime(s.day)
        fut = i in ("nq", "es")
        if i == "spyfwd":
            if S3.get("spy") is None:
                continue
            base = S3["spy"].copy(); base["day"] = pd.to_datetime(base.day)
            fitd, sc = base, s
        else:
            fitd, sc = s[s.day <= DEV_END], s[s.day > DEV_END]
        for zid, yname, px, vx in (("Z1", "dayr", ["a5", "pdr", "onr"], ["rv_pd"] + (["rv_on"] if fut else [])),
                                   ("Z2", "rest15", ["a5", "pdr", "onr", "fh15"], ["rv_15"])):
            cols = [yname] + px + vx
            a = fitd.dropna(subset=cols); b = sc.dropna(subset=cols)
            if len(b) < 30:
                continue
            def X(z, v):
                return np.column_stack([np.ones(len(z))] + [lg(z[c]) for c in px + (vx if v else [])])
            ya, yb = lg(a[yname]), lg(b[yname])
            b0, _ = ols_cl(X(a, False), ya, np.arange(len(a)))
            b1, se1 = ols_cl(X(a, True), ya, np.arange(len(a)))
            bh, seh = ols_cl(X(b, True), yb, np.arange(len(b)))
            g = 1 - np.mean((yb - X(b, True) @ b1) ** 2) / np.mean((yb - X(b, False) @ b0) ** 2)
            k = len(vx)
            same = all(np.sign(b1[-k + j]) == np.sign(bh[-k + j]) for j in range(k))
            co = "; ".join(f"{vx[j]} {b1[-k+j]:+.3f} ({b1[-k+j]/se1[-k+j]:+.1f}) / {bh[-k+j]:+.3f} ({bh[-k+j]/seh[-k+j]:+.1f})" for j in range(k))
            L.append(f"| {i} | {zid} | {len(a):,} / {len(b):,} | {g*100:+.2f}% | {co} | {'**PASS**' if (g >= 0.02 and same) else 'fail'} |")
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
