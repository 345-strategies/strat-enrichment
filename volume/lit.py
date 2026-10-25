"""Volume study round 5 (PREREG_ROUND5_LITERATURE.md): LIT1 Gao et al. intraday momentum by volume,
LIT2a/b Campbell-Grossman-Wang volume and autocorrelation (daily, 30m), LIT3 Bulkowski throwbacks.
Usage: python lit.py <work dir> <out.md> [inst ...]"""
import os, sys
import numpy as np
import pandas as pd
from analyze import ols_cl, DEV_END


def fit(X, y, cl):
    b, se = ols_cl(X, y, cl)
    return b, b / se


def verdict(sign, bd, bh, th):
    if np.sign(bd) == sign and np.sign(bh) == sign and abs(th) >= 2:
        return "holds"
    if np.sign(bd) == sign and np.sign(bh) == sign:
        return "partly"
    return "no"


def main():
    w, outp = sys.argv[1], sys.argv[2]
    L = ["# Volume study round 5: literature claims on our data (PREREG_ROUND5_LITERATURE.md)\n"]
    insts = sys.argv[3:] or ["nq", "es", "spy"]
    # LIT1
    L.append("## LIT1 Gao et al. (2018): first half-hour return predicts the last half-hour, more so on high-volume days\n")
    L.append("| inst | part | days | slope r1 (t) | R² | slope high-vol tercile (t) | slope low-vol tercile (t) | high minus low (t) | timing bp/day high / low |\n|---|---|---|---|---|---|---|---|---|")
    res1 = {}
    for i in insts:
        p = f"{w}/s4_{i}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p).dropna(subset=["r1", "r12", "r13", "rv_30"]).copy()
        d["day"] = pd.to_datetime(d.day)
        dv = d.day <= DEV_END
        lo, hi = np.quantile(d.loc[dv, "rv_30"], [1 / 3, 2 / 3])
        d["g"] = np.where(d.rv_30 >= hi, 1, np.where(d.rv_30 <= lo, -1, 0))
        for nm, m in (("dev", dv), ("holdout", ~dv)):
            x = d[m]
            cl = np.arange(len(x))
            b, t = fit(np.c_[np.ones(len(x)), x.r1, x.r12], x.r13.values, cl)
            pred = np.c_[np.ones(len(x)), x.r1, x.r12] @ b
            r2 = 1 - np.var(x.r13 - pred) / np.var(x.r13)
            sl = {}
            for gg in (1, -1):
                z = x[x.g == gg]
                bz, tz = fit(np.c_[np.ones(len(z)), z.r1, z.r12], z.r13.values, np.arange(len(z)))
                sl[gg] = (bz[1], tz[1], (np.sign(z.r1) * z.r13).mean() * 1e4)
            z = x[x.g != 0]
            hh = (z.g == 1).astype(float).values
            bi, ti = fit(np.c_[np.ones(len(z)), z.r1, z.r12, hh, z.r1 * hh], z.r13.values, np.arange(len(z)))
            res1[(i, nm)] = (b[1], t[1], bi[4], ti[4])
            L.append(f"| {i} | {nm} | {len(x)} | {b[1]:+.4f} ({t[1]:+.1f}) | {r2*100:.1f}% | {sl[1][0]:+.4f} ({sl[1][1]:+.1f}) | {sl[-1][0]:+.4f} ({sl[-1][1]:+.1f}) | {bi[4]:+.4f} ({ti[4]:+.1f}) | {sl[1][2]:+.2f} / {sl[-1][2]:+.2f} |")
    L.append("")
    for i in insts:
        if (i, "dev") in res1:
            a, b = res1[(i, "dev")], res1[(i, "holdout")]
            L.append(f"- {i}: base effect (slope > 0) **{verdict(1, a[0], b[0], b[1])}**; stronger on high volume (interaction > 0) **{verdict(1, a[2], b[2], b[3])}**.")
    # LIT2a
    L.append("\n## LIT2a Campbell, Grossman, Wang (1993): daily autocorrelation lower after high-volume days\n")
    L.append("| inst | part | pairs | b0 r(t) (t) | b1 r(t) x v(t) (t) |\n|---|---|---|---|---|")
    res2 = {}
    for i in insts:
        p = f"{w}/s4_{i}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p).sort_values("si").reset_index(drop=True)
        d["day"] = pd.to_datetime(d.day)
        nxt = d.set_index("si").rd
        d["rn"] = d.si.map(lambda s: nxt.get(s + 1, np.nan))
        d = d[(d.nextok == 1)].dropna(subset=["rd", "rn", "vdt"])
        dv = d.day <= DEV_END
        for nm, m in (("dev", dv), ("holdout", ~dv)):
            x = d[m]
            b, t = fit(np.c_[np.ones(len(x)), x.rd, x.rd * x.vdt, x.vdt], x.rn.values, np.arange(len(x)))
            res2[(i, nm)] = (b[2], t[2])
            L.append(f"| {i} | {nm} | {len(x)} | {b[1]:+.3f} ({t[1]:+.1f}) | {b[2]:+.3f} ({t[2]:+.1f}) |")
    L.append("")
    for i in insts:
        if (i, "dev") in res2:
            a, b = res2[(i, "dev")], res2[(i, "holdout")]
            L.append(f"- {i}: interaction < 0 **{verdict(-1, a[0], b[0], b[1])}**.")
    # LIT2b
    L.append("\n## LIT2b intraday version: consecutive 30m RTH bars, r(k+1) on r(k) and r(k) x log RVOL(k), session-clustered\n")
    L.append("| inst | part | pairs | b0 r(k) (t) | b1 r(k) x log RVOL (t) |\n|---|---|---|---|---|")
    res3 = {}
    for i in insts:
        p = f"{w}/s5_{i}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p).dropna()
        d["day"] = pd.to_datetime(d.day)
        dv = d.day <= DEV_END
        for nm, m in (("dev", dv), ("holdout", ~dv)):
            x = d[m]
            lv = np.log(np.maximum(x.rv0.values, 0.05))
            b, t = fit(np.c_[np.ones(len(x)), x.r0, x.r0 * lv, lv], x.r1.values, x.si.values)
            res3[(i, nm)] = (b[2], t[2])
            L.append(f"| {i} | {nm} | {len(x):,} | {b[1]:+.3f} ({t[1]:+.1f}) | {b[2]:+.3f} ({t[2]:+.1f}) |")
    L.append("")
    for i in insts:
        if (i, "dev") in res3:
            a, b = res3[(i, "dev")], res3[(i, "holdout")]
            L.append(f"- {i}: interaction < 0 **{verdict(-1, a[0], b[0], b[1])}**.")
    # LIT3
    L.append("\n## LIT3 Bulkowski: high-volume breakouts throw back to the breakout level more often\n")
    L.append("| inst | part | S1 throwback rate top / bottom rv_trig tercile | difference (t) | S2 PDH/PDL throwback top / bottom | difference (t) |\n|---|---|---|---|---|---|")
    res4 = {}
    for i in insts:
        p1, p2 = f"{w}/s1_{i}.parquet", f"{w}/s2_{i}.parquet"
        if not os.path.exists(p1):
            continue
        t1 = pd.read_parquet(p1); t1 = t1[t1.risk_atr >= 0.05].reset_index(drop=True); t1["day"] = pd.to_datetime(t1.day)
        t2 = pd.read_parquet(p2).reset_index(drop=True); t2["day"] = pd.to_datetime(t2.day)
        d1, d2 = t1.day <= DEV_END, t2.day <= DEV_END
        g1 = np.zeros(len(t1), int)
        for k in (15, 30, 60):
            mk = (t1.tf == k).values
            lo, hi = np.quantile(t1.loc[mk & d1.values, "rv_trig"], [1 / 3, 2 / 3])
            v = t1.loc[mk, "rv_trig"].values
            g1[mk] = np.where(v >= hi, 1, np.where(v <= lo, -1, 0))
        lo, hi = np.quantile(t2.loc[d2, "rv_trig"], [1 / 3, 2 / 3])
        g2 = np.where(t2.rv_trig >= hi, 1, np.where(t2.rv_trig <= lo, -1, 0))
        for nm, m1, m2 in (("dev", d1.values, d2.values), ("holdout", ~d1.values, ~d2.values)):
            a = t1.tb.values.astype(float); b = t2.tb2.values.astype(float)
            ga = np.where(m1, g1, 0); gb = np.where(m2, g2, 0)
            Xa = (ga != 0)
            ba, ta = fit(np.c_[np.ones(Xa.sum()), (ga[Xa] == 1)], a[Xa], t1.si.values[Xa])
            Xb = (gb != 0)
            bb, tb_ = fit(np.c_[np.ones(Xb.sum()), (gb[Xb] == 1)], b[Xb], t2.si.values[Xb])
            res4[(i, nm)] = (ba[1], ta[1], bb[1], tb_[1])
            L.append(f"| {i} | {nm} | {a[ga==1].mean():.3f} / {a[ga==-1].mean():.3f} | {ba[1]*100:+.1f} pts ({ta[1]:+.1f}) | {b[gb==1].mean():.3f} / {b[gb==-1].mean():.3f} | {bb[1]*100:+.1f} pts ({tb_[1]:+.1f}) |")
    L.append("")
    for i in insts:
        if (i, "dev") in res4:
            a, b = res4[(i, "dev")], res4[(i, "holdout")]
            L.append(f"- {i}: S1 throwbacks higher on high volume **{verdict(1, a[0], b[0], b[1])}**; S2 **{verdict(1, a[2], b[2], b[3])}**.")
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
