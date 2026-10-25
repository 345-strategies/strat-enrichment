"""Volume study round 2 (PREREG_ROUND2.md, exploratory): E1 range with strong price controls, E2 range-based
exit policy, E3 signed volume vs unweighted, E4 big-day classifier.
Usage: python analyze2.py <dir with s1_/s3_*.parquet> <out.md> [inst ...]  (spyfwd is scored with the SPY dev+holdout fit)"""
import sys
import numpy as np
import pandas as pd
from analyze import ols_cl, DEV_END

LF = 0.05


def lg(x):
    return np.log(np.maximum(np.asarray(x, float), LF))


def design(x, vol):
    lb = pd.cut(x.mins_left, [-1, 30, 60, 120, 180, 270, 400], labels=False).values
    cols = [np.ones(len(x)), lg(x.c1rng), lg(x.sdrng), lg(x.a5), lg(x.pdr), lg(x.onr)]
    cols += [(lb == b).astype(float) for b in range(1, 6)]
    cols += [(x.tf.values == 30).astype(float), (x.tf.values == 60).astype(float)]
    if vol:
        cols += [lg(x.rv_c1), lg(x.rv_day), lg(x.rv_trig)]
    return np.column_stack(cols)


VOLN = ["rv_c1", "rv_day", "rv_trig"]


def prep(t):
    t = t[(t.risk_atr >= 0.05) & (t.rng_rest > 0)].reset_index(drop=True).copy()
    t["day"] = pd.to_datetime(t["day"])
    return t


def e1_e2(dv, ho, inst, L, fitpart="dev"):
    y_d, y_h = np.log(dv.rng_rest.values), np.log(ho.rng_rest.values)
    b0, _ = ols_cl(design(dv, False), y_d, dv.si.values)
    b1, se1 = ols_cl(design(dv, True), y_d, dv.si.values)
    bh, seh = ols_cl(design(ho, True), y_h, ho.si.values) if len(ho) > 50 else (b1 * np.nan, se1 * np.nan)
    p0, p1 = design(ho, False) @ b0, design(ho, True) @ b1
    gain = 1 - np.mean((y_h - p1) ** 2) / np.mean((y_h - p0) ** 2)
    k = len(VOLN)
    same = all(np.sign(b1[-k + i]) == np.sign(bh[-k + i]) for i in range(k))
    coefs = "; ".join(f"{e} {fitpart} {b1[-k+i]:+.3f} (t {b1[-k+i]/se1[-k+i]:+.1f}) holdout {bh[-k+i]:+.3f} (t {bh[-k+i]/seh[-k+i]:+.1f})" for i, e in enumerate(VOLN))
    L.append(f"| {inst} | E1 | {len(dv):,} / {len(ho):,} | {gain*100:+.2f}% | {'yes' if same else 'no'} | {coefs} | {'**PASS**' if (gain >= 0.01 and same) else 'fail'} |")
    # E2 policy
    rows = []
    for nm, x, P0, P1 in (("fit", dv, design(dv, False) @ b0, design(dv, True) @ b1), ("holdout", ho, p0, p1)):
        rows.append((nm, x, P0, P1))
    m0 = np.median(design(dv, False) @ b0); m1 = np.median(design(dv, True) @ b1)
    out = {}
    for nm, x, P0, P1 in rows:
        r0 = np.where(P0 > m0, x.R.values, x.R1R.values)
        r1 = np.where(P1 > m1, x.R.values, x.R1R.values)
        d = r1 - r0
        bb, se = ols_cl(np.ones((len(x), 1)), d, x.si.values)
        out[nm] = dict(hold=x.R.mean(), t1r=x.R1R.mean(), rp=r0.mean(), rv=r1.mean(), diff=bb[0], t=bb[0] / se[0] if se[0] > 0 else np.nan, n=len(x))
    return gain, same, out


def e3(t, dev):
    res = {}
    for nm, m in (("dev", dev), ("holdout", ~dev)):
        x = t[m & t.hit1.notna()]
        X = np.column_stack([np.ones(len(x)), x.su_c1, x.sv_c1, (x.tf == 30).astype(float), (x.tf == 60).astype(float)])
        b, se = ols_cl(X, x.hit1.values, x.si.values)
        Xu = np.column_stack([np.ones(len(x)), x.su_c1, (x.tf == 30).astype(float), (x.tf == 60).astype(float)])
        bu, seu = ols_cl(Xu, x.hit1.values, x.si.values)
        res[nm] = dict(bsv=b[2], tsv=b[2] / se[2], bsu=b[1], tsu=b[1] / se[1], bsu_only=bu[1], tsu_only=bu[1] / seu[1], n=len(x))
    return res


def e4(s, dev):
    s = s[(s.rest > 0)].copy()
    y = (s.rest.values >= 1.0).astype(float)
    def X(x, vol):
        c = [np.ones(len(x)), lg(x.fh), lg(x.a5), lg(x.pdr), lg(x.onr)]
        if vol:
            c.append(lg(x.rv_fh))
        return np.column_stack(c)
    d = dev.values
    b0, _ = ols_cl(X(s[d], False), y[d], s[d].si.values)
    b1, se1 = ols_cl(X(s[d], True), y[d], s[d].si.values)
    bh, seh = ols_cl(X(s[~d], True), y[~d], s[~d].si.values) if (~d).sum() > 30 else (b1 * np.nan, se1 * np.nan)
    p0 = np.clip(X(s[~d], False) @ b0, 0, 1); p1 = np.clip(X(s[~d], True) @ b1, 0, 1)
    br0 = np.mean((y[~d] - p0) ** 2); br1 = np.mean((y[~d] - p1) ** 2)
    return dict(n_dev=int(d.sum()), n_ho=int((~d).sum()), base=y[~d].mean(), gain=1 - br1 / br0, b=b1[-1], t=b1[-1] / se1[-1], bh=bh[-1], th=bh[-1] / seh[-1])


def main():
    d, outp = sys.argv[1], sys.argv[2]
    insts = sys.argv[3:] or ["nq", "es", "spy", "spyfwd"]
    L = ["# Volume study round 2 (exploratory, PREREG_ROUND2.md)\n",
         "R net of costs. t is session-clustered. Fit on dev (to 2022-12-31), scored on holdout (2023+). SPY forward: fit on all SPY 2018-05..2026-04-21, scored on Tradier 2026-05-08..2026-10-19 (untouched).\n",
         "\n## E1 rest-of-session range: does volume add to a strong price model?\n",
         "| inst | test | n fit / scored | MSE gain from volume | coefs keep sign | volume coefficients | PASS |\n|---|---|---|---|---|---|---|"]
    E2, E3, E4 = [], [], []
    T = {i: prep(pd.read_parquet(f"{d}/s1_{i}.parquet")) for i in insts if __import__("os").path.exists(f"{d}/s1_{i}.parquet")}
    S = {i: pd.read_parquet(f"{d}/s3_{i}.parquet") for i in insts if __import__("os").path.exists(f"{d}/s3_{i}.parquet")}
    for i, t in T.items():
        if i == "spyfwd":
            if "spy" not in T:
                continue
            g, same, o = e1_e2(T["spy"], t, i, L, fitpart="fit(all SPY)")
        else:
            dev = t.day <= DEV_END
            g, same, o = e1_e2(t[dev], t[~dev], i, L)
            E3.append((i, e3(t, dev)))
        E2.append((i, o))
    L.append("\n## E2 exit policy: hold to close when predicted range is high, else +1R target\n")
    L.append("| inst | part | n | fixed hold | fixed +1R | rule, price model | rule, price + volume | volume minus price | t |\n|---|---|---|---|---|---|---|---|---|")
    for i, o in E2:
        for nm, r in o.items():
            L.append(f"| {i} | {nm} | {r['n']:,} | {r['hold']:+.4f} | {r['t1r']:+.4f} | {r['rp']:+.4f} | {r['rv']:+.4f} | {r['diff']:+.4f} | {r['t']:+.2f} |")
    L.append("\n## E3 signed volume vs unweighted signed share, P(+1R first)\n")
    L.append("| inst | part | n | su_c1 alone (t) | su_c1 with sv_c1 (t) | sv_c1 added (t) |\n|---|---|---|---|---|---|")
    for i, r in E3:
        for nm, x in r.items():
            L.append(f"| {i} | {nm} | {x['n']:,} | {x['bsu_only']:+.4f} ({x['tsu_only']:+.1f}) | {x['bsu']:+.4f} ({x['tsu']:+.1f}) | {x['bsv']:+.4f} ({x['tsv']:+.1f}) |")
    L.append("\n## E4 big-day classifier at 10:30 ET (rest of session range >= 1 ATR20)\n")
    L.append("| inst | sessions dev / holdout | holdout base rate | Brier gain from volume | rv_fh coef dev (t) | holdout (t) | PASS |\n|---|---|---|---|---|---|---|")
    for i, s in S.items():
        s = s.copy(); s["day"] = pd.to_datetime(s["day"])
        if i == "spyfwd":
            if "spy" not in S:
                continue
            ss = pd.concat([S["spy"].assign(day=pd.to_datetime(S["spy"]["day"]), part=1), s.assign(part=0)], ignore_index=True)
            ss["si"] = np.arange(len(ss))
            r = e4(ss, ss.part == 1)
        else:
            r = e4(s, s.day <= DEV_END)
        ok = r["gain"] >= 0.01 and np.sign(r["b"]) == np.sign(r["bh"])
        L.append(f"| {i} | {r['n_dev']:,} / {r['n_ho']:,} | {r['base']:.3f} | {r['gain']*100:+.2f}% | {r['b']:+.3f} ({r['t']:+.1f}) | {r['bh']:+.3f} ({r['th']:+.1f}) | {'**PASS**' if ok else 'fail'} |")
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
