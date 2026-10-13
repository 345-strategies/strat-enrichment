"""Pull the numbers we compare against from The Strat Stats (NQ) page.

The page (by notprofessorgreen on X) embeds its data as `var D={...}`. This
keeps only what the comparison tab needs: info, baselines, the 4x4 setup cells
and the FTC split. Usage: python prep_nq_compare.py <saved page.html> <out.json>
"""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
line = max(s.split("\n"), key=len)
i = line.index("var D=") + len("var D=")
D, _ = json.JSONDecoder().raw_decode(line, i)
X = D["syms"]["NQ"]
F = ['n', 'cb', 'exp', 'cth', 'stop', 'res', 'p1', 'p2', 'mfe', 'mae', 'ratr', 'dcb', 'd1', 'qcb', 'q1',
     'fbThis', 'fbOther', 'fbNone', 'fbBase', 'rpct']
cells = []
for tf, rows in X["rows"].items():
    for key, a in rows.items():
        c2, c1, d = key.split("|")
        r = dict(zip(F, a))
        cells.append(dict(tf=tf, dir="long" if d == "1" else "short", c2t=c2, c1t=c1,
                          n=r["n"], p1=r["p1"], p2=r["p2"], cb=r["cb"], d1=r["d1"], q1=r["q1"]))
fc = X["T"]["ftc"]["c"]
ftc = [dict(zip(fc, r)) for r in X["T"]["ftc"]["r"]]
ftc = [dict(tf=r["tf"], dir=r["dir"], ftc=r["ftc"], n=r["n"], p1=r["p1R"], d1=r["d_1R"], q1=r["q_1R"]) for r in ftc]
mc = X["T"]["meta"]["c"]
meta = [dict(zip(mc, r)) for r in X["T"]["meta"]["r"]]
out = dict(info=X["info"], tfs=X["tfs"], base=X["base"], cells=cells, ftc=ftc, meta=meta,
           source="https://claude.ai/artifact/JQ5Uam7QrbktqqrAd54ZoY")
json.dump(out, open(dst, "w"), separators=(",", ":"))
print(dst, len(cells), "cells", len(ftc), "ftc rows")
