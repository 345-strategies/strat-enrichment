"""Replication check: our atlas run on NQ (his window, his timeframes) vs the NQ Strat Stats page.

Usage: python compare_nq.py <atlas_nq.json> <nq_compare.json> <out.json>
Writes a compact summary (meta, baselines, cell agreement, significant cells) for the page.
"""
import json, sys
import numpy as np

ours, his, dst = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
MIN_N = 100
tfs = [t for t in his["tfs"] if t in ours["info"]["trade_tfs"]]

meta = []
om = {x["tf"]: x for x in ours["meta"]}
for h in his["meta"]:
    o = om.get(h["tf"])
    if o:
        meta.append(dict(tf=h["tf"], his_bars=h["bars"], our_bars=o["bars"], his_events=h["events"], our_events=o["events"],
                         his_amb=h["ambiguous"], our_amb=o["ambiguous"]))

ob = {(x["tf"], x["dir"]): x for x in ours["base"]}
base = []
for tf in tfs:
    for d in ("long", "short"):
        h, o = his["base"][tf][d], ob[(tf, d)]
        base.append(dict(tf=tf, dir=d, his_p1=h["p1"], our_p1=o["p1"], his_p2=h["p2"], our_p2=o["p2"],
                         his_cb=h["cb"], our_cb=o["cb"], his_res=h["res"], our_res=o["res"]))

oc = {(x["tf"], x["dir"], x["c2t"], x["c1t"]): x for x in ours["setups_type"]}
pairs = []
for h in his["cells"]:
    o = oc.get((h["tf"], h["dir"], h["c2t"], h["c1t"]))
    if o and h["n"] >= MIN_N and o["n"] >= MIN_N:
        pairs.append(dict(tf=h["tf"], dir=h["dir"], c2=h["c2t"], c1=h["c1t"], his_n=h["n"], our_n=int(o["n"]),
                          his_d1=h["d1"], our_d1=o["d1"], his_q1=h["q1"], our_q1=o["q1"], his_p1=h["p1"], our_p1=o["p1"]))
hd = np.array([p["his_d1"] for p in pairs]); od = np.array([p["our_d1"] for p in pairs])
nr = np.array([p["our_n"] / p["his_n"] for p in pairs])
agree = dict(cells=len(pairs), sign_match=round(float((np.sign(hd) == np.sign(od)).mean() * 100), 1),
             corr=round(float(np.corrcoef(hd, od)[0, 1]), 3), mean_abs_diff=round(float(np.abs(hd - od).mean()), 2),
             within_2pts=round(float((np.abs(hd - od) <= 2).mean() * 100), 1),
             n_ratio_median=round(float(np.median(nr)), 3), n_ratio_p10_p90=[round(float(x), 3) for x in np.percentile(nr, [10, 90])])
by_tf = []
for tf in tfs:
    sel = [p for p in pairs if p["tf"] == tf]
    if len(sel) > 2:
        a = np.array([p["his_d1"] for p in sel]); b = np.array([p["our_d1"] for p in sel])
        by_tf.append(dict(tf=tf, cells=len(sel), corr=round(float(np.corrcoef(a, b)[0, 1]), 3),
                          sign_match=round(float((np.sign(a) == np.sign(b)).mean() * 100), 1)))
sig_his = [p for p in pairs if p["his_q1"] is not None and p["his_q1"] < 0.05]
sig_ours = [p for p in pairs if p["our_q1"] is not None and p["our_q1"] < 0.05]
out = dict(info=ours["info"], meta=meta, base=base, agree=agree, by_tf=by_tf, sig_his=sig_his, sig_ours=sig_ours, cells=pairs)
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(dict(meta=meta, agree=agree, by_tf=by_tf, sig_his=sig_his, sig_ours=sig_ours), indent=1))
for b in base:
    print(b)
