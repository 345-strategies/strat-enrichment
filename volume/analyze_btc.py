"""Volume study round 7: the same tests on Bitcoin (PREREG_ROUND7_BTC.md).
Usage: python analyze_btc.py <work dir> <out.md>
Needs s1-s3 for btc (Binance BTCUSDT, primary) and btccb (Coinbase BTC-USD, second venue); btcfwd is optional.
The literature tests run separately: python lit.py <work dir> <out.md> btc btccb"""
import os, sys
import numpy as np
import pandas as pd
from analyze import ols_cl, run_inst, DEV_END
import analyze2 as A2
import analyze3 as A3

LF = 0.05
PRIM = [f"H{i}" for i in range(1, 10)]


def rd(p):
    return pd.read_parquet(p) if os.path.exists(p) else None


def lg(x):
    return np.log(np.maximum(np.asarray(x, float), LF))


def mean_t(y, cl):
    m = np.isfinite(y)
    b, se = ols_cl(np.ones((m.sum(), 1)), y[m], cl[m])
    return b[0], b[0] / se[0] if se[0] > 0 else np.nan


def cdiff(y, g, cl):
    m = (g != 0) & np.isfinite(y)
    if (g[m] == 1).sum() < 5 or (g[m] == -1).sum() < 5:
        return np.nan, np.nan, 0, 0
    X = np.c_[np.ones(m.sum()), (g[m] == 1).astype(float)]
    b, se = ols_cl(X, y[m], cl[m])
    return b[1], b[1] / se[1], int((g[m] == 1).sum()), int((g[m] == -1).sum())


def main():
    w, outp = sys.argv[1], sys.argv[2]
    L = ["# Volume study round 7: Bitcoin (PREREG_ROUND7_BTC.md)\n",
         "btc = Binance BTCUSDT spot (primary), btccb = Coinbase BTC-USD (second venue). Sessions are UTC days. "
         "R is net of 0.10% round trip. t is session-clustered. Dev to 2022-12-31, holdout 2023 on.\n"]
    S1 = {i: rd(f"{w}/s1_{i}.parquet") for i in ("btc", "btccb", "btcfwd")}
    S2 = {i: rd(f"{w}/s2_{i}.parquet") for i in ("btc", "btccb", "btcfwd")}
    S3 = {i: rd(f"{w}/s3_{i}.parquet") for i in ("btc", "btccb", "btcfwd")}
    main2 = [i for i in ("btc", "btccb") if S1[i] is not None]

    # ---------------- direction (round 1 tests)
    fam, rng = [], []
    for i in main2:
        r, rr = run_inst(S1[i], S2[i], i, L, fam)
        rng.extend(rr)
    F = pd.DataFrame(fam)
    F.to_csv(outp.replace(".md", "_contrasts.csv"), index=False)
    P = F[F.hid.isin(PRIM)]
    L.append("\n## Direction H1-H9\n")
    L.append("| inst | H | test | part | A minus B | t | perm p | nA | nB | mean A | mean B |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in P.iterrows():
        L.append(f"| {r.inst} | {r.hid} | {r.desc} | {r.part} | {r['diff']:+.4f} | {r.t:+.2f} | {r.perm_p:.3f} | {r.nA:,} | {r.nB:,} | {r.mA:+.4f} | {r.mB:+.4f} |")
    bh = P[(P.inst == "btc") & (P.part == "holdout")].set_index("hid")
    pv = bh.perm_p.reindex(PRIM).values
    holm = np.full(len(PRIM), np.nan); run = 0
    for rank, i in enumerate(np.argsort(pv)):
        v = min(1, pv[i] * (len(PRIM) - rank)) if np.isfinite(pv[i]) else np.nan
        run = max(run, v) if np.isfinite(v) else run
        holm[i] = run
    L.append("\nPass rule: Binance same sign dev/holdout, Binance holdout |t| >= 2, Holm perm p < 0.05, Coinbase holdout same sign.\n")
    L.append("| H | btc same sign | btc holdout t | Holm p | btccb holdout same sign | PASS |\n|---|---|---|---|---|---|")
    npass = 0
    for j, h in enumerate(PRIM):
        g = lambda inst, part: P[(P.inst == inst) & (P.part == part) & (P.hid == h)]
        bd, bho, ch = g("btc", "dev"), g("btc", "holdout"), g("btccb", "holdout")
        if not len(bd) or not len(bho):
            continue
        bd, bho = bd.iloc[0], bho.iloc[0]
        ss = np.sign(bd["diff"]) == np.sign(bho["diff"])
        tt = abs(bho.t) >= 2
        cb = len(ch) and np.sign(ch.iloc[0]["diff"]) == np.sign(bho["diff"])
        ok = bool(ss and tt and holm[j] < 0.05 and cb)
        npass += ok
        L.append(f"| {h} | {'yes' if ss else 'no'} | {bho.t:+.2f} | {holm[j]:.3f} | {'yes' if cb else 'no'} | {'**PASS**' if ok else 'fail'} |")
    for part in ("dev", "holdout"):
        x = P[P.part == part]
        L.append(f"\nFamily view, {part}: {int((x.t.abs() >= 2).sum())} of {int(x.t.notna().sum())} contrasts reach |t| >= 2 (about {0.0455 * x.t.notna().sum():.1f} by chance).")
    L.append(f"\n**Direction: {npass} of 9 pass on BTC.**\n")
    L.append("\n## Secondary contrasts (not used for the verdict)\n")
    L.append("| inst | id | test | part | A minus B | t | nA | nB |\n|---|---|---|---|---|---|---|---|")
    for _, r in F[~F.hid.isin(PRIM)].iterrows():
        L.append(f"| {r.inst} | {r.hid} | {r.desc} | {r.part} | {r['diff']:+.4f} | {r.t:+.2f} | {r.nA:,} | {r.nB:,} |")
    # cost check (diagnostic, not in the pre-registration's pass rules): fees are a share of price, so a small-range
    # C1 pays more R in fees; volume tracks range, so net-R contrasts could pick up fees. Same tests on gross R.
    L.append("\n## Fee check: H1-H9 on gross R (before the 0.10% fee), diagnostic only\n")
    L.append("| inst | part | fee in R, median | " + " | ".join(PRIM) + " |\n|---|---|---|" + "---|" * len(PRIM))
    for i in main2:
        t = S1[i].copy(); fee = 0.001 * t.ent / t.risk
        t["R"] = t.R + fee
        if "R1R" in t:
            t["R1R"] = t.R1R + fee
        gf = []
        run_inst(t, S2[i], i, [], gf)
        G = pd.DataFrame(gf)
        for part, dm in (("dev", pd.to_datetime(t.day) <= DEV_END), ("holdout", pd.to_datetime(t.day) > DEV_END)):
            x = G[(G.part == part)].set_index("hid")
            L.append(f"| {i} | {part} | {np.median(fee[dm.values]):.3f} | " + " | ".join(
                f"{x.loc[h, 'diff']:+.4f} ({x.loc[h, 't']:+.1f})" if h in x.index else "" for h in PRIM) + " |")
    L.append("\n## Range R1-R2 (rest-of-session range after a break, holdout MSE gain)\n")
    L.append("| inst | test | holdout gain | coefs keep sign | coefficients |\n|---|---|---|---|---|")
    for r in rng:
        L.append(f"| {r['inst']} | {r['rid']} | {r['gain']*100:+.2f}% | {'yes' if r['same'] else 'no'} | {r['coefs']} |")

    # ---------------- round 2
    L.append("\n## E1 range with strong price controls\n")
    L.append("| inst | test | n fit / scored | MSE gain from volume | coefs keep sign | volume coefficients | PASS |\n|---|---|---|---|---|---|---|")
    E2, E3 = [], []
    T = {i: A2.prep(S1[i]) for i in main2}
    for i, t in T.items():
        dev = t.day <= DEV_END
        _, _, o = A2.e1_e2(t[dev], t[~dev], i, L)
        E2.append((i, o)); E3.append((i, A2.e3(t, dev)))
    if S1["btcfwd"] is not None and "btc" in T:
        _, _, o = A2.e1_e2(T["btc"], A2.prep(S1["btcfwd"]), "btcfwd", L, fitpart="fit(all btc)")
        E2.append(("btcfwd", o))
    L.append("\n## E2 exit policy: hold to close when predicted range is high, else +1R\n")
    L.append("| inst | part | n | fixed hold | fixed +1R | rule, price | rule, price + volume | volume minus price | t |\n|---|---|---|---|---|---|---|---|---|")
    for i, o in E2:
        for nm, r in o.items():
            L.append(f"| {i} | {nm} | {r['n']:,} | {r['hold']:+.4f} | {r['t1r']:+.4f} | {r['rp']:+.4f} | {r['rv']:+.4f} | {r['diff']:+.4f} | {r['t']:+.2f} |")
    L.append("\n## E3 signed volume vs unweighted signed share, P(+1R first)\n")
    L.append("| inst | part | n | su_c1 alone (t) | su_c1 with sv_c1 (t) | sv_c1 added (t) |\n|---|---|---|---|---|---|")
    for i, r in E3:
        for nm, x in r.items():
            L.append(f"| {i} | {nm} | {x['n']:,} | {x['bsu_only']:+.4f} ({x['tsu_only']:+.1f}) | {x['bsu']:+.4f} ({x['tsu']:+.1f}) | {x['bsv']:+.4f} ({x['tsv']:+.1f}) |")
    L.append("\n## E4 big-day call at 01:00 UTC (rest of day >= 1 ATR20)\n")
    L.append("| inst | sessions dev / holdout | holdout base rate | Brier gain from volume | coef dev (t) | holdout (t) |\n|---|---|---|---|---|---|")
    for i in main2:
        s = S3[i].copy(); s["day"] = pd.to_datetime(s["day"])
        r = A2.e4(s, s.day <= DEV_END)
        L.append(f"| {i} | {r['n_dev']:,} / {r['n_ho']:,} | {r['base']:.3f} | {r['gain']*100:+.2f}% | {r['b']:+.3f} ({r['t']:+.1f}) | {r['bh']:+.3f} ({r['th']:+.1f}) |")

    # ---------------- round 3 leads on BTC
    SIGN = {"L1": -1, "L2": -1, "L3": 1, "L4": 1}
    a, b = [], []
    for i in main2:
        x = S1[i][S1[i].risk_atr >= 0.05].copy(); x["inst"] = i
        x["part"] = np.where(pd.to_datetime(x.day) <= DEV_END, "dev", "ho"); a.append(x)
        y = S2[i].copy(); y["inst"] = i
        y["part"] = np.where(pd.to_datetime(y.day) <= DEV_END, "dev", "ho"); b.append(y)
    a, b = pd.concat(a, ignore_index=True), pd.concat(b, ignore_index=True)
    Hall = A3.leads(a, b, ["inst", "part"])
    Hho = A3.leads(a[a.part == "ho"], b[b.part == "ho"], ["inst"])
    L.append("\n## Round-3 leads on BTC (Binance + Coinbase pooled)\n")
    L.append("| lead | all years: A-B (t) | holdout: A-B (t) | n A / n B holdout | expected sign | confirmed |\n|---|---|---|---|---|---|")
    for k in ("L1", "L2", "L3", "L4"):
        d0, t0, _, _ = Hall[k]; d1, t1, na, nb = Hho[k]
        ok = np.isfinite(t1) and np.sign(d1) == SIGN[k] and abs(t1) >= 1.65
        L.append(f"| {k} | {d0:+.4f} ({t0:+.2f}) | {d1:+.4f} ({t1:+.2f}) | {na} / {nb} | {'+' if SIGN[k] > 0 else '-'} | {'**yes**' if ok else 'no'} |")

    # ---------------- round 4 M (management) and Z (sizing)
    L.append("\n## M. Management on 1m, mean per-trade difference in net R\n")
    L.append("| inst | part | n | M1 | M1b | M2 | M2b | M3 | M3b | M4 |\n|---|---|---|---|---|---|---|---|---|---|")
    MR = {}
    for i in ("btc", "btccb", "btcfwd"):
        t = S1[i]
        if t is None:
            continue
        t = t[t.risk_atr >= 0.05].copy().reset_index(drop=True); t["day"] = pd.to_datetime(t.day)
        splits = [("forward", np.ones(len(t), bool))] if i == "btcfwd" else [("dev", (t.day <= DEV_END).values), ("holdout", (t.day > DEV_END).values)]
        for nm, m in splits:
            x = t[m].reset_index(drop=True)
            cl = x.day.dt.strftime("%Y%m%d").values
            cells = [mean_t(c.values, cl) for c in (x.R_cxw - x.R_pxw, x.R_cxw - x.R, x.R_cxa - x.R_pxa, x.R_cxa - x.R)]
            def gl(f):
                g = np.zeros(len(x), int)
                for k in (15, 30, 60):
                    mk = (x.tf == k).values
                    if not mk.any():
                        continue
                    ref = t[(t.tf == k) & ((t.day <= DEV_END) if i != "btcfwd" else True)][f].values
                    lo, hi = np.nanquantile(ref, [1 / 3, 2 / 3])
                    v = x.loc[mk, f].values
                    g[mk] = np.where(v >= hi, 1, np.where(v <= lo, -1, 0))
                return g
            gt, gd = gl("rv_trig"), gl("rv_day")
            for col, g in (((x.R_w15 - x.R).values, gt), ((x.R_t05 - x.R).values, gt), ((x.R - x.R1R).values, gd)):
                d, tt, _, _ = cdiff(col, g, cl); cells.append((d, tt))
            MR[(i, nm)] = cells
            L.append(f"| {i} | {nm} | {len(x):,} | " + " | ".join(f"{bb:+.4f} ({tt:+.1f})" for bb, tt in cells) + " |")
    L.append("\nPass rule: same sign dev/holdout on btc and btccb, holdout |t| >= 2 on both, same sign forward (if present).\n")
    L.append("| test | btc dev / holdout | btccb dev / holdout | forward | PASS |\n|---|---|---|---|---|")
    for j, n in enumerate(["M1", "M1b", "M2", "M2b", "M3", "M3b", "M4"]):
        try:
            ed, eh = MR[("btc", "dev")][j], MR[("btc", "holdout")][j]
            sd, sh = MR[("btccb", "dev")][j], MR[("btccb", "holdout")][j]
        except KeyError:
            continue
        fw = MR.get(("btcfwd", "forward"), [(np.nan, np.nan)] * 7)[j]
        sg = np.sign(eh[0])
        ok = (np.sign(ed[0]) == sg and np.sign(sd[0]) == sg and np.sign(sh[0]) == sg and abs(eh[1]) >= 2 and abs(sh[1]) >= 2
              and (not np.isfinite(fw[0]) or np.sign(fw[0]) == sg))
        L.append(f"| {n} | {ed[0]:+.4f} / {eh[0]:+.4f} (t {eh[1]:+.1f}) | {sd[0]:+.4f} / {sh[0]:+.4f} (t {sh[1]:+.1f}) | {fw[0]:+.4f} (t {fw[1]:+.1f}) | {'**PASS**' if ok else 'fail'} |")
    L.append("\n## Z. Sizing: does volume forecast the day's range? (holdout MSE gain of log range vs price-only)\n")
    L.append("| inst | test | sessions fit / scored | MSE gain from volume | volume coef dev (t) / holdout (t) |\n|---|---|---|---|---|")
    for i in main2:
        s = S3[i].copy(); s["day"] = pd.to_datetime(s.day)
        fitd, sc = s[s.day <= DEV_END], s[s.day > DEV_END]
        for zid, yname, px, vx in (("Z1", "dayr", ["a5", "pdr"], ["rv_pd"]), ("Z2", "rest15", ["a5", "pdr", "fh15"], ["rv_15"])):
            cols = [yname] + px + vx
            a_, b_ = fitd.dropna(subset=cols), sc.dropna(subset=cols)
            if len(b_) < 30:
                continue
            X = lambda z, v: np.column_stack([np.ones(len(z))] + [lg(z[c]) for c in px + (vx if v else [])])
            ya, yb = lg(a_[yname]), lg(b_[yname])
            b0, _ = ols_cl(X(a_, False), ya, np.arange(len(a_)))
            b1, se1 = ols_cl(X(a_, True), ya, np.arange(len(a_)))
            bh, seh = ols_cl(X(b_, True), yb, np.arange(len(b_)))
            g = 1 - np.mean((yb - X(b_, True) @ b1) ** 2) / np.mean((yb - X(b_, False) @ b0) ** 2)
            L.append(f"| {i} | {zid} | {len(a_):,} / {len(b_):,} | {g*100:+.2f}% | {b1[-1]:+.3f} ({b1[-1]/se1[-1]:+.1f}) / {bh[-1]:+.3f} ({bh[-1]/seh[-1]:+.1f}) |")
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
