"""Volume study 2026-10-24: the pre-registered analysis (PREREG.md + amendment A).
Usage: python analyze.py <dir with s1_*.parquet, s2_*.parquet> <out.md> [inst ...]"""
import sys, math
import numpy as np
import pandas as pd

DEV_END = pd.Timestamp("2022-12-31")
RNG = np.random.default_rng(20261024)
NPERM = 2000
FEATS = ["rv_c1", "rv_trig", "rv_c2", "rv_day", "sv_c1"]


def pnorm2(t):
    return math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else float("nan")


def ols_cl(X, y, cl):
    """OLS with session-clustered SE. Returns beta, se."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    codes, inv = np.unique(cl, return_inverse=True)
    G = len(codes)
    sc = np.zeros((G, X.shape[1]))
    np.add.at(sc, inv, X * u[:, None])
    meat = sc.T @ sc
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k) if G > 1 else 1
    V = XtX @ meat @ XtX * adj
    return b, np.sqrt(np.diag(V))


def contrast(y, g, cl):
    """g: +1 group A, -1 group B, 0 ignore. Returns diff A-B, clustered t, nA, nB, meanA, meanB."""
    m = (g != 0) & np.isfinite(y)
    if m.sum() < 10 or (g[m] == 1).sum() < 5 or (g[m] == -1).sum() < 5:
        return dict(diff=np.nan, t=np.nan, nA=int((g[m] == 1).sum()), nB=int((g[m] == -1).sum()), mA=np.nan, mB=np.nan)
    X = np.c_[np.ones(m.sum()), (g[m] == 1).astype(float)]
    b, se = ols_cl(X, y[m], cl[m])
    return dict(diff=b[1], t=b[1] / se[1] if se[1] > 0 else np.nan, nA=int((g[m] == 1).sum()), nB=int((g[m] == -1).sum()),
                mA=float(y[m][g[m] == 1].mean()), mB=float(y[m][g[m] == -1].mean()))


def perm_p(y, g, block, obs):
    m = (g != 0) & np.isfinite(y)
    y, g, block = y[m], g[m], block[m]
    if len(y) < 10 or not np.isfinite(obs):
        return np.nan
    order = np.argsort(block, kind="stable")
    y, g, block = y[order], g[order], block[order]
    _, starts = np.unique(block, return_index=True)
    bounds = list(starts) + [len(y)]
    cnt = 0
    for _ in range(NPERM):
        gp = g.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            gp[a:b] = RNG.permutation(gp[a:b])
        d = y[gp == 1].mean() - y[gp == -1].mean()
        cnt += abs(d) >= abs(obs) - 1e-12
    return (cnt + 1) / (NPERM + 1)


def terc_labels(t, f, keys, dev):
    """+1 top dev tercile, -1 bottom, 0 middle; cut points from dev per key group."""
    g = np.zeros(len(t), int)
    for kv, idx in t.groupby(keys[0] if len(keys) == 1 else keys).groups.items():
        sub = t.loc[idx]
        dsub = sub[dev.loc[idx]]
        lo, hi = np.nanquantile(dsub[f], [1 / 3, 2 / 3])
        pos = t.index.get_indexer(idx)
        g[pos] = np.where(sub[f] >= hi, 1, np.where(sub[f] <= lo, -1, 0))
    return g


def q_labels(t, f, keys, dev, q, top):
    """+1 if in the dev-quantile extreme (top: >= q quantile; else <= q quantile), -1 otherwise."""
    g = np.zeros(len(t), int)
    for kv, idx in t.groupby(keys[0] if len(keys) == 1 else keys).groups.items():
        sub = t.loc[idx]
        c = np.nanquantile(sub[dev.loc[idx]][f], q)
        pos = t.index.get_indexer(idx)
        g[pos] = np.where(sub[f] >= c, 1, -1) if top else np.where(sub[f] <= c, 1, -1)
    return g


def run_inst(t, s2, inst, lines, fam):
    t = t[t.risk_atr >= 0.05].reset_index(drop=True)
    t["day"] = pd.to_datetime(t["day"])
    dev = t.day <= DEV_END
    yr = t.day.dt.year.values
    block = t.tf.astype(str).values + "_" + yr.astype(str)
    cl = t.si.values
    lines.append(f"\n## {inst.upper()}\n")
    lines.append(f"S1 trades after the 0.05 ATR stop floor: {len(t):,} (dev {int(dev.sum()):,}, holdout {int((~dev).sum()):,}); "
                 f"sessions {t.si.nunique():,}; {t.day.min().date()} .. {t.day.max().date()}.\n")
    # baseline context
    lines.append("Baseline (all S1 trades, no volume filter), mean net R to stop-or-close and P(+1R first):\n")
    lines.append("| TF | part | n | mean R | clustered t | P(+1R first) |\n|---|---|---|---|---|---|")
    for k in (15, 30, 60):
        for nm, msk in (("dev", dev), ("holdout", ~dev)):
            s = t[(t.tf == k) & msk]
            b, se = ols_cl(np.ones((len(s), 1)), s.R.values, s.si.values)
            lines.append(f"| {k}m | {nm} | {len(s):,} | {b[0]:+.3f} | {b[0]/se[0]:+.1f} | {np.nanmean(s.hit1):.3f} |")
    # labels
    G = {f: terc_labels(t, f, ["tf"], dev) for f in FEATS}
    res = []

    def do(hid, desc, y, g, msk):
        for nm, part in (("dev", dev.values & msk), ("holdout", (~dev.values) & msk)):
            gg = np.where(part, g, 0)
            c = contrast(y, gg, cl)
            p = perm_p(y, gg, block, c["diff"]) if nm == "holdout" or hid.startswith("H") else np.nan
            res.append(dict(inst=inst, hid=hid, desc=desc, part=nm, **c, perm_p=p))

    allm = np.ones(len(t), bool)
    R = t.R.values; H1 = t.hit1.values
    do("H1", "rv_c1 high vs low tercile, R", R, G["rv_c1"], allm)
    do("H2", "rv_trig high vs low tercile, R", R, G["rv_trig"], allm)
    do("H3", "rv_c2 high vs low tercile, R", R, G["rv_c2"], allm)
    do("H4", "rv_day high vs low tercile, R", R, G["rv_day"], allm)
    do("H5", "sv_c1 high vs low tercile, R", R, G["sv_c1"], allm)
    do("H6", "rv_c1 high vs low, TFC-aligned only, R", R, G["rv_c1"], t.tfc.values == 1)
    dry = q_labels(t, "rv_c1", ["tf"], dev, 0.25, top=False)
    do("H7", "inside C1: dry-up vs other inside, R", R, dry, t.t1.values == 1)
    clx = q_labels(t, "rv_c1", ["tf"], dev, 0.90, top=True)
    do("H8", "2-2 reversal: climax C1 vs other, R", R, clx, t.t1.values == -2)
    # secondary on R/hit1 (not used for verdict)
    for f in FEATS:
        do("s_hit_" + f, f"{f} high vs low tercile, P(+1R first)", H1, G[f], allm)
    for k in (15, 30, 60):
        for f in FEATS:
            do(f"s_{k}_{f}", f"{k}m only: {f} high vs low, R", R, G[f], t.tf.values == k)
    is212 = (t.t1.values == 1) & (np.abs(t.t0.values) == 2)
    is312 = (t.t1.values == 1) & (t.t0.values == 3)
    do("s_212", "2-1-2: rv_c1 high vs low, R", R, G["rv_c1"], is212)
    do("s_312", "3-1-2: rv_c1 high vs low, R", R, G["rv_c1"], is312)
    do("s_tfc_trig", "TFC-aligned: rv_trig high vs low, R", R, G["rv_trig"], t.tfc.values == 1)
    # S2
    s2 = s2.copy(); s2["day"] = pd.to_datetime(s2["day"])
    d2 = s2.day <= DEV_END
    lines.append(f"\nS2 PDH/PDL breaks: {len(s2):,} (dev {int(d2.sum()):,}, holdout {int((~d2).sum()):,}). "
                 f"P(expand) dev {s2[d2].expand.mean():.3f}, holdout {s2[~d2].expand.mean():.3f}.\n")
    g9 = terc_labels(s2.assign(k=1), "rv_trig", ["k"], d2)
    g9d = terc_labels(s2.assign(k=1), "rv_day", ["k"], d2)
    b2 = s2.day.dt.year.astype(str).values
    for hid, desc, g, y in (("H9", "PDH/PDL break: rv_trig high vs low, P(expand)", g9, s2.expand.values.astype(float)),
                            ("s_s2_move", "PDH/PDL break: rv_trig high vs low, move to close (ATR)", g9, s2.move.values),
                            ("s_s2_day", "PDH/PDL break: rv_day high vs low, P(expand)", g9d, s2.expand.values.astype(float))):
        for nm, part in (("dev", d2.values), ("holdout", ~d2.values)):
            gg = np.where(part, g, 0)
            c = contrast(y, gg, s2.si.values)
            p = perm_p(y, gg, b2, c["diff"])
            res.append(dict(inst=inst, hid=hid, desc=desc, part=nm, **c, perm_p=p))
    fam.extend(res)
    # range family
    rr = []
    tt = t[(t.rng_rest > 0)].copy()
    ok = np.ones(len(tt), bool)
    for c in ("c1rng", "sdrng", "rv_c1", "rv_day", "rv_trig"):
        v = tt[c].values.astype(float)
        ok &= np.isfinite(v) & (v > 0)
    if (~ok).any():   # never on futures or stocks; crypto has a few zero-range / zero-volume bars
        lines.append(f"\nRange family: {int((~ok).sum()):,} rows with a zero or missing range/volume input left out.\n")
        tt = tt[ok].copy()
    tt["lb"] = pd.cut(tt.mins_left, [-1, 30, 60, 120, 180, 270, 400], labels=False)
    def design(x, extra):
        cols = [np.ones(len(x)), np.log(x.c1rng), np.log(x.sdrng)]
        cols += [(x.lb == b).astype(float) for b in range(1, 6)]
        cols += [(x.tf == 30).astype(float), (x.tf == 60).astype(float)]
        cols += [np.log(x[e]) for e in extra]
        return np.column_stack(cols)
    dv = tt.day <= DEV_END
    y = np.log(tt.rng_rest.values)
    for rid, extra in (("R1", ["rv_c1", "rv_day"]), ("R2", ["rv_trig"])):
        Xd0, Xh0 = design(tt[dv], []), design(tt[~dv], [])
        Xd1, Xh1 = design(tt[dv], extra), design(tt[~dv], extra)
        b0, _ = ols_cl(Xd0, y[dv.values], tt[dv].si.values)
        b1, se1 = ols_cl(Xd1, y[dv.values], tt[dv].si.values)
        bh, seh = ols_cl(Xh1, y[~dv.values], tt[~dv].si.values)
        mse0 = np.mean((y[~dv.values] - Xh0 @ b0) ** 2)
        mse1 = np.mean((y[~dv.values] - Xh1 @ b1) ** 2)
        k = len(extra)
        coefs = "; ".join(f"{e} dev {b1[-k+i]:+.3f} (t {b1[-k+i]/se1[-k+i]:+.1f}) holdout {bh[-k+i]:+.3f} (t {bh[-k+i]/seh[-k+i]:+.1f})"
                          for i, e in enumerate(extra))
        same = all(np.sign(b1[-k + i]) == np.sign(bh[-k + i]) for i in range(k))
        gain = 1 - mse1 / mse0
        rr.append(dict(inst=inst, rid=rid, gain=gain, same=same, coefs=coefs, passed=bool(gain >= 0.01 and same)))
    return res, rr


def main():
    d, outp = sys.argv[1], sys.argv[2]
    insts = sys.argv[3:] or ["nq", "es", "spy"]
    lines = ["# Volume study results (pre-registered family)\n", "Generated by analyze.py. R is net of costs. Contrasts are group A minus group B; t is session-clustered; perm p is the within-TF-x-year permutation null (2,000 draws).\n"]
    fam, rng = [], []
    for inst in insts:
        try:
            t = pd.read_parquet(f"{d}/s1_{inst}.parquet"); s2 = pd.read_parquet(f"{d}/s2_{inst}.parquet")
        except FileNotFoundError:
            lines.append(f"\n## {inst.upper()}: no data file\n"); continue
        r, rr = run_inst(t, s2, inst, lines, fam)
        rng.extend(rr)
    F = pd.DataFrame(fam)
    F.to_csv(outp.replace(".md", "_contrasts.csv"), index=False)
    prim = [f"H{i}" for i in range(1, 10)]
    lines.append("\n## Primary family H1-H9\n")
    lines.append("| inst | H | test | part | A minus B | t | perm p | nA | nB | mean A | mean B |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in F[F.hid.isin(prim)].iterrows():
        lines.append(f"| {r.inst} | {r.hid} | {r.desc} | {r.part} | {r['diff']:+.4f} | {r.t:+.2f} | {r.perm_p:.3f} | {r.nA:,} | {r.nB:,} | {r.mA:+.4f} | {r.mB:+.4f} |")
    # pass rule
    lines.append("\n## Pass rule per hypothesis\n")
    lines.append("| H | primary NQ: same sign dev/holdout | NQ holdout |t|>=2 | NQ holdout perm p (Holm) | ES holdout same sign | SPY holdout same sign | PASS |\n|---|---|---|---|---|---|---|")
    P = F[F.hid.isin(prim)]
    nqh = P[(P.inst == "nq") & (P.part == "holdout")].set_index("hid")
    pv = nqh.perm_p.reindex(prim)
    order = np.argsort(pv.values)
    holm = np.full(len(prim), np.nan)
    run = 0
    for rank, i in enumerate(order):
        v = min(1, pv.values[i] * (len(prim) - rank)) if np.isfinite(pv.values[i]) else np.nan
        run = max(run, v) if np.isfinite(v) else run
        holm[i] = run
    for i, h in enumerate(prim):
        def get(inst, part):
            x = P[(P.inst == inst) & (P.part == part) & (P.hid == h)]
            return x.iloc[0] if len(x) else None
        nd, nh = get("nq", "dev"), get("nq", "holdout")
        eh, sh = get("es", "holdout"), get("spy", "holdout")
        if nd is None or nh is None:
            continue
        ss = np.sign(nd["diff"]) == np.sign(nh["diff"])
        tt = abs(nh.t) >= 2
        hp = holm[i] < 0.05
        es_ok = eh is not None and np.sign(eh["diff"]) == np.sign(nh["diff"])
        sp_ok = sh is not None and np.sign(sh["diff"]) == np.sign(nh["diff"])
        lines.append(f"| {h} | {'yes' if ss else 'no'} | {'yes' if tt else 'no'} ({nh.t:+.2f}) | {holm[i]:.3f} | {'yes' if es_ok else 'no'} | {'yes' if sp_ok else 'no'} | {'**PASS**' if (ss and tt and hp and es_ok and sp_ok) else 'fail'} |")
    dv = P[P.part == "dev"]
    lines.append(f"\nFamily view: {int((dv.t.abs() >= 2).sum())} of {int(dv.t.notna().sum())} dev contrasts reach |t| >= 2 (about {0.0455*dv.t.notna().sum():.1f} expected by chance); "
                 f"holdout: {int((P[P.part=='holdout'].t.abs() >= 2).sum())} of {int(P[P.part=='holdout'].t.notna().sum())}.\n")
    lines.append("\n## Range / regime family R1-R2\n")
    lines.append("| inst | test | holdout MSE gain | volume coefs keep sign | coefficients | PASS |\n|---|---|---|---|---|---|")
    for r in rng:
        lines.append(f"| {r['inst']} | {r['rid']} | {r['gain']*100:+.2f}% | {'yes' if r['same'] else 'no'} | {r['coefs']} | {'**PASS**' if r['passed'] else 'fail'} |")
    lines.append("\n## Secondary contrasts (not used for the verdict)\n")
    lines.append("| inst | id | test | part | A minus B | t | perm p | nA | nB |\n|---|---|---|---|---|---|---|---|---|")
    for _, r in F[~F.hid.isin(prim)].iterrows():
        lines.append(f"| {r.inst} | {r.hid} | {r.desc} | {r.part} | {r['diff']:+.4f} | {r.t:+.2f} | {r.perm_p:.3f} | {r.nA:,} | {r.nB:,} |")
    open(outp, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:200]))


if __name__ == "__main__":
    main()
