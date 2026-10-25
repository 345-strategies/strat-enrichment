"""Volume study round 8: the same tests on single stocks (PREREG_ROUND8_STOCKS.md).
Usage: python analyze_stocks.py <work dir> <out.md> NAME [NAME ...]
Reads s1/s2 for each stock, runs the round-1 family on each and on all of them pooled (sessions clustered by
date across stocks), on net R and on gross R (before the percentage cost)."""
import os, sys
import numpy as np
import pandas as pd
from analyze import run_inst, DEV_END

PRIM = [f"H{i}" for i in range(1, 10)]


def holm(pv):
    out = np.full(len(pv), np.nan); run = 0
    for rank, i in enumerate(np.argsort(pv)):
        v = min(1, pv[i] * (len(pv) - rank)) if np.isfinite(pv[i]) else np.nan
        run = max(run, v) if np.isfinite(v) else run
        out[i] = run
    return out


def gross(t, cost_pct):
    t = t.copy(); fee = cost_pct * t.ent / t.risk
    t["R"] = t.R + fee
    if "R1R" in t:
        t["R1R"] = t.R1R + fee
    return t


def main():
    w, outp, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    cp = float(os.environ.get("COST_PCT", "0.0002"))
    S1 = {n: pd.read_parquet(f"{w}/s1_{n}.parquet") for n in names}
    S2 = {n: pd.read_parquet(f"{w}/s2_{n}.parquet") for n in names}
    # pooled: one cluster per date across stocks
    a = pd.concat([S1[n].assign(inst=n) for n in names], ignore_index=True)
    b = pd.concat([S2[n].assign(inst=n) for n in names], ignore_index=True)
    days = pd.Index(sorted(set(pd.to_datetime(a.day)) | set(pd.to_datetime(b.day))))
    a["si"] = days.get_indexer(pd.to_datetime(a.day)); b["si"] = days.get_indexer(pd.to_datetime(b.day))
    S1["pooled"], S2["pooled"] = a, b
    L = ["# Volume study round 8: single stocks (PREREG_ROUND8_STOCKS.md)\n",
         f"Stocks: {', '.join(names)}; pooled = all of them, t clustered by date. Net R is after {cp * 1e4:.0f} bp round trip; "
         "gross R is before it. Dev to 2022-12-31, holdout 2023 on.\n"]
    fam, rng = [], []
    for basis in ("net", "gross"):
        for n in names + ["pooled"]:
            t = S1[n] if basis == "net" else gross(S1[n], cp)
            f = []
            r, rr = run_inst(t, S2[n], n, L if basis == "net" else [], f)
            for x in f:
                x["basis"] = basis
            fam.extend(f)
            if basis == "net":
                rng.extend(rr)
    F = pd.DataFrame(fam)
    F.to_csv(outp.replace(".md", "_contrasts.csv"), index=False)
    P = F[F.hid.isin(PRIM)]
    g = lambda inst, part, h, basis="net": P[(P.inst == inst) & (P.part == part) & (P.hid == h) & (P.basis == basis)]
    L.append("\n## Direction H1-H9, pooled\n")
    L.append("| H | test | basis | dev diff (t) | holdout diff (t) | holdout perm p |\n|---|---|---|---|---|---|")
    for h in PRIM:
        for basis in ("net", "gross"):
            d, o = g("pooled", "dev", h, basis), g("pooled", "holdout", h, basis)
            if len(d) and len(o):
                d, o = d.iloc[0], o.iloc[0]
                L.append(f"| {h} | {o.desc} | {basis} | {d['diff']:+.4f} ({d.t:+.2f}) | {o['diff']:+.4f} ({o.t:+.2f}) | {o.perm_p:.3f} |")
    hp = holm(np.array([g("pooled", "holdout", h).perm_p.iloc[0] if len(g("pooled", "holdout", h)) else np.nan for h in PRIM]))
    L.append("\nPass rule: pooled net same sign dev/holdout, pooled net holdout |t| >= 2, Holm perm p < 0.05, pooled gross "
             "holdout same sign with |t| >= 2, and at least 4 of 5 stocks with the same holdout sign (net).\n")
    L.append("| H | same sign | net holdout t | Holm p | gross holdout t | stocks agreeing | PASS |\n|---|---|---|---|---|---|---|")
    npass = 0
    for j, h in enumerate(PRIM):
        d, o, og = g("pooled", "dev", h), g("pooled", "holdout", h), g("pooled", "holdout", h, "gross")
        if not (len(d) and len(o) and len(og)):
            continue
        d, o, og = d.iloc[0], o.iloc[0], og.iloc[0]
        sg = np.sign(o["diff"])
        agree = sum(int(len(g(n, "holdout", h)) and np.sign(g(n, "holdout", h).iloc[0]["diff"]) == sg) for n in names)
        ok = bool(np.sign(d["diff"]) == sg and abs(o.t) >= 2 and hp[j] < 0.05 and np.sign(og["diff"]) == sg and abs(og.t) >= 2
                  and agree >= max(1, len(names) - 1))
        npass += ok
        L.append(f"| {h} | {'yes' if np.sign(d['diff']) == sg else 'no'} | {o.t:+.2f} | {hp[j]:.3f} | {og.t:+.2f} | {agree} of {len(names)} | {'**PASS**' if ok else 'fail'} |")
    for part in ("dev", "holdout"):
        x = P[(P.part == part) & (P.basis == "net") & (P.inst != "pooled")]
        L.append(f"\nFamily view, {part}, per stock, net: {int((x.t.abs() >= 2).sum())} of {int(x.t.notna().sum())} contrasts reach |t| >= 2 (about {0.0455 * x.t.notna().sum():.1f} by chance).")
    L.append(f"\n**Direction: {npass} of 9 pass on stocks.**\n")
    L.append("\n## Per stock, holdout, net (diff, t)\n")
    L.append("| H | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for h in PRIM:
        L.append(f"| {h} | " + " | ".join(f"{g(n, 'holdout', h).iloc[0]['diff']:+.4f} ({g(n, 'holdout', h).iloc[0].t:+.1f})" if len(g(n, "holdout", h)) else "" for n in names) + " |")
    L.append("\n## Secondary contrasts, pooled (not used for the verdict)\n")
    L.append("| id | test | basis | part | A minus B | t | nA | nB | mean A | mean B |\n|---|---|---|---|---|---|---|---|---|---|")
    for _, r in F[(~F.hid.isin(PRIM)) & (F.inst == "pooled")].iterrows():
        L.append(f"| {r.hid} | {r.desc} | {r.basis} | {r.part} | {r['diff']:+.4f} | {r.t:+.2f} | {r.nA:,} | {r.nB:,} | {r.mA:+.4f} | {r.mB:+.4f} |")
    L.append("\n## Range R1-R2 (rest-of-session range after a break, holdout MSE gain)\n")
    L.append("| inst | test | holdout gain | coefs keep sign | coefficients |\n|---|---|---|---|---|")
    for r in rng:
        L.append(f"| {r['inst']} | {r['rid']} | {r['gain'] * 100:+.2f}% | {'yes' if r['same'] else 'no'} | {r['coefs']} |")
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-60:]))


if __name__ == "__main__":
    main()
