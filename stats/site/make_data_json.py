"""The page's results as one JSON file for agents and anyone pulling the numbers.

Usage: python make_data_json.py <atlas_es.json> <nq_replication.json> <out.json> <updated YYYY-MM-DD>
"""
import json
import sys

atlas, rep, dst, updated = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
rep.pop("cells", None)
out = {
    "about": "Strat Stats results. Field guide: https://345strategies.com/research/strat-stats.md (Definitions). "
             "p1/p2 = % of resolved first breaks reaching +1R/+2R before the stop; d1 = p1 minus the same timeframe+direction "
             "baseline, in points; q1 = Benjamini-Hochberg q; n = trades; c2,c1 = combo bars (C1 is the setup); "
             "cb = trigger bar closes beyond entry; f2 = trigger bar is a failed 2 (broke only the setup bar's high or low, "
             "then closed back inside it).",
    "source": "https://345strategies.com/research/strat-stats",
    "code": "https://github.com/345-strategies/strat-enrichment",
    "credit": "Format after The Strat Stats by @notprofgreen, https://x.com/notprofgreen",
    "updated": updated,
    "es": atlas,
    "nq_replication": rep,
}
open(dst, "w").write(json.dumps(out, separators=(",", ":")))
print(dst)
