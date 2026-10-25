"""Volume study round 3 (PREREG_ROUND3.md): pooled round-1 estimates of four leads, and the forward check
on Tradier SPY+QQQ 2026-05-08..2026-10-19. Usage: python analyze3.py <work dir> <out.md>"""
import os, sys, math
import numpy as np
import pandas as pd
from analyze import ols_cl


def labels(t, f, keys, mode):
    g = np.zeros(len(t), int)
    for _, idx in t.groupby(keys).groups.items():
        pos = t.index.get_indexer(idx)
        x = t.loc[idx, f].values
        if mode == "terc":
            lo, hi = np.nanquantile(x, [1 / 3, 2 / 3])
            g[pos] = np.where(x >= hi, 1, np.where(x <= lo, -1, 0))
        else:
            g[pos] = np.where(x >= np.nanquantile(x, 0.9), 1, -1)
    return g


def contrast(y, g, cl):
    m = (g != 0) & np.isfinite(y)
    if (g[m] == 1).sum() < 5 or (g[m] == -1).sum() < 5:
        return np.nan, np.nan, int((g[m] == 1).sum()), int((g[m] == -1).sum())
    X = np.c_[np.ones(m.sum()), (g[m] == 1).astype(float)]
    b, se = ols_cl(X, y[m], cl[m])
    return b[1], b[1] / se[1], int((g[m] == 1).sum()), int((g[m] == -1).sum())


def leads(s1, s2, part_keys):
    """s1, s2 already carry inst; terciles within inst x part x tf (S1) or inst x part (S2)."""
    out = {}
    s1 = s1.reset_index(drop=True); s2 = s2.reset_index(drop=True)
    cl1 = pd.to_datetime(s1.day).dt.strftime("%Y%m%d").values
    cl2 = pd.to_datetime(s2.day).dt.strftime("%Y%m%d").values
    g = labels(s2, "rv_trig", part_keys, "terc")
    out["L1"] = contrast(s2.expand.values.astype(float), g, cl2)
    sub = s1[s1.tfc == 1].reset_index(drop=True)
    g = labels(sub, "rv_c1", part_keys + ["tf"], "terc")
    out["L2"] = contrast(sub.R.values, g, pd.to_datetime(sub.day).dt.strftime("%Y%m%d").values)
    g = labels(s1, "rv_c1", part_keys + ["tf"], "terc")
    out["L3"] = contrast(s1.R.values, g, cl1)
    sub = s1[s1.t1 == -2].reset_index(drop=True)
    g = labels(sub, "rv_c1", part_keys + ["tf"], "dec")
    out["L4"] = contrast(sub.R.values, g, pd.to_datetime(sub.day).dt.strftime("%Y%m%d").values)
    return out


def main():
    w, outp = sys.argv[1], sys.argv[2]
    SIGN = {"L1": -1, "L2": -1, "L3": 1, "L4": 1}
    DESC = {"L1": "PDH/PDL break: high vs low trigger RVOL, P(expand)", "L2": "TFC-aligned breaks: high vs low C1 RVOL, R",
            "L3": "all breaks: high vs low C1 RVOL, R", "L4": "2-2 reversals: climax C1 vs other, R"}
    def load(insts):
        a = []; b = []
        for i in insts:
            if os.path.exists(f"{w}/s1_{i}.parquet"):
                x = pd.read_parquet(f"{w}/s1_{i}.parquet"); x = x[x.risk_atr >= 0.05]; x["inst"] = i
                x["part"] = np.where(pd.to_datetime(x.day) <= pd.Timestamp("2022-12-31"), "dev", "ho")
                a.append(x)
                y = pd.read_parquet(f"{w}/s2_{i}.parquet"); y["inst"] = i
                y["part"] = np.where(pd.to_datetime(y.day) <= pd.Timestamp("2022-12-31"), "dev", "ho")
                b.append(y)
        return pd.concat(a, ignore_index=True), pd.concat(b, ignore_index=True)
    L = ["# Volume study round 3: forward check of four leads (PREREG_ROUND3.md)\n"]
    h1, h2 = load(["nq", "es", "spy"])
    f1, f2 = load(["spyfwd", "qqqfwd"])
    H = leads(h1, h2, ["inst", "part"])
    F = leads(f1, f2, ["inst"])
    L.append(f"Forward data: {f1.groupby('inst').day.nunique().to_dict()} sessions with trades; S1 trades {len(f1):,}, S2 breaks {len(f2):,}; "
             f"{pd.to_datetime(f1.day).min().date()} .. {pd.to_datetime(f1.day).max().date()}.\n")
    L.append("| lead | test | round-1 pooled (3 inst, both halves): A-B (t) | forward SPY+QQQ: A-B (t) | n A / n B forward | expected sign | confirmed |\n|---|---|---|---|---|---|---|")
    for k in ("L1", "L2", "L3", "L4"):
        hd, ht, _, _ = H[k]; fd, ft, na, nb = F[k]
        ok = np.isfinite(ft) and np.sign(fd) == SIGN[k] and abs(ft) >= 1.65
        L.append(f"| {k} | {DESC[k]} | {hd:+.4f} ({ht:+.2f}) | {fd:+.4f} ({ft:+.2f}) | {na} / {nb} | {'+' if SIGN[k] > 0 else '-'} | {'**yes**' if ok else 'no'} |")
    # forward baseline context
    L.append("\nForward baseline (no volume filter): " + "; ".join(
        f"{i} {k}m n {len(g):,} mean R {g.R.mean():+.3f}" for (i, k), g in f1.groupby(["inst", "tf"])))
    L.append("; S2 P(expand) " + "; ".join(f"{i} {g.expand.mean():.3f} (n {len(g)})" for i, g in f2.groupby("inst")) + "\n")
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
