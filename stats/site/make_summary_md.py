"""Plain-text twin of the Strat Stats page for search engines, answer engines and agents.

Usage: python make_summary_md.py <atlas_es.json> <nq_replication.json> <out.md>
"""
import json, sys

a, rep, dst = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
info = a["info"]
tested = [c for c in a["setups"] if c["n"] >= info["min_n"] and c.get("q1") is not None]
sig = sorted([c for c in tested if c["q1"] < 0.05], key=lambda c: c["q1"])
base = {(b["tf"], b["dir"]): b for b in a["base"]}
A = rep["agree"]
his = rep["sig_his"][0]

L = []
w = L.append
w("# Strat Stats: ES and NQ futures, 2018-2026")
w("")
w("Descriptive statistics for TheStrat candlestick setups on CME E-mini S&P 500 (ES) futures, built from "
  f"{info['minutes']:,} one-minute bars ({info['start']} to {info['end']}), with a replication on Nasdaq-100 (NQ) futures. "
  "Published by 345 Strategies (https://345strategies.com). Format after The Strat Stats by @notprofgreen (https://x.com/notprofgreen). "
  "Research-only: not a strategy, not a forecast, not financial advice.")
w("")
w("Interactive page: https://345strategies.com/research/strat-stats")
w("Machine-readable results (JSON): https://345strategies.com/research/strat-stats.json")
w("")
w("## Key findings")
w("")
w(f"- The setup label alone barely beats a random break. Of {len(tested)} setup cells with at least {info['min_n']} trades, "
  f"{len(sig)} differ from their baseline at q < 0.05 (Benjamini-Hochberg).")
w("- Across every first break, +1R is reached before the stop about 49-54% of the time, depending on timeframe and direction.")
d = next(c for c in a["setups"] if c["tf"] == "15m" and c["dir"] == "long" and c["c2"] == "2D" and c["c1"] == "2D")
w(f"- The one cell flagged on both ES and NQ is a negative: 15m 2D then 2D, traded long ({d['d1']:+.1f} points vs baseline on ES; "
  f"{his['our_d1']:+.1f} on NQ in our run, {his['his_d1']:+.1f} on @notprofgreen's page).")
w("- Timeframe continuity (TFC) adds nothing measurable to these first breaks.")
w(f"- Replication: run on NQ over @notprofgreen's window ({rep['info']['start']} to {rep['info']['end']}) and timeframes, the same code "
  f"reproduces his page: bar counts within 0.1%, {A['cells']} setup cells with lift correlation {A['corr']:.2f}, "
  f"same sign {A['sign_match']:.0f}% of the time, mean difference {A['mean_abs_diff']} points. His only significant cell, "
  f"15m 2D-2D long, flags too ({his['our_d1']:+.1f} vs his {his['his_d1']:+.1f}).")
w("- What it means: the edge, if any, is not in the bar label. It has to come from context, targets, or management, which this study does not test.")
w("")
# Questions people ask answer engines, answered from the same numbers. make_site_page.py turns this section into FAQPage JSON-LD.
p1 = [b["p1"] for b in a["base"] if b["tf"] in info["trade_tfs"]]
p2 = [b["p2"] for b in a["base"] if b["tf"] in info["trade_tfs"]]
f2 = [c for c in sig if c["c1"].startswith("F2")]
f2_list = "; ".join(f"{c['tf']} {c['c2']} then {c['c1']}, traded {c['dir']} ({c['d1']:+.1f} points)" for c in f2)
f2_worse = sum(c["d1"] < 0 for c in f2)
w("## Questions")
w("")
w("### Does The Strat work?")
w("")
w(f"Not from the bar label alone, on this test. Of {len(tested)} TheStrat setup cells on ES futures with at least {info['min_n']} trades, "
  f"{len(sig)} differ from a random break at q < 0.05. The test is one entry (the first break of the setup bar) and one 1R bracket. "
  "Context, targets and trade management are not tested, so this does not say whether a trader using TheStrat makes money.")
w("")
w("### What is TheStrat's win rate?")
w("")
w(f"There is no single number. Across every first break on ES futures from {info['start'][:4]} to {info['end'][:4]}, "
  f"{min(p1):.0f}-{max(p1):.0f}% reach +1R before the stop and {min(p2):.0f}-{max(p2):.0f}% reach +2R, depending on timeframe "
  f"({', '.join(info['trade_tfs'])}) and direction. The {len(sig)} combos that differ from that baseline at q < 0.05 move +1R by "
  f"{min(abs(c['d1']) for c in sig):.0f}-{max(abs(c['d1']) for c in sig):.0f} points.")
w("")
w("### Do failed 2s work better?")
w("")
w(f"Not as a group. {len(f2)} of the {len(sig)} setup cells that differ from baseline have a failed 2 as the setup bar: {f2_list}. "
  f"{f2_worse} of the {len(f2)} do worse than a random break.")
w("")
w("## Definitions")
w("")
w("- Bar types compare each bar with the one before, highs and lows only. 1 = inside (broke neither side). 2U / 2D = broke the prior high only / low only. "
  "3U / 3D = broke both; 3U if it closed above its own open, otherwise 3D. Equal is not a break.")
w("- Failed 2 (F2U / F2D): a 2 that closes back inside the prior bar's range, edges included. Only 2s can fail; there is no F3.")
w("- Combo: two closed bars, C2 then C1. C1 is the setup bar.")
w("- Entry bar: the bar after C1 closes. Long = first trade through C1's high, short = first trade through C1's low (a stop order; "
  "if the minute gaps through, the fill is that minute's open). No break, no trade. A minute that breaks both sides is dropped.")
w("- Stop: the other side of C1 plus one tick. 1R = entry to stop.")
w("- Outcome: +1R (and +2R) reached before the stop within the entry bar plus two more bars, checked on 1-minute bars. "
  "The entry minute is skipped; a minute touching stop and target counts as the stop.")
w("- Baseline: all first breaks on the same timeframe and direction. Wilson 95% intervals, z-tests, Benjamini-Hochberg q-values.")
w("- Sessions: 18:00 ET (15:00 PT) start; 4h bars at 18, 22, 02, 06, 10, 14 ET. Prices back-adjusted at each contract roll from raw prices.")
w("")
w("## Baselines (ES, all first breaks)")
w("")
w("| Timeframe | Direction | Trades | +1R before stop | +2R before stop | Trigger bar closes beyond entry |")
w("|---|---|---|---|---|---|")
for tf in info["trade_tfs"]:
    for d in ("long", "short"):
        b = base[(tf, d)]
        w(f"| {tf} | {d} | {int(b['n']):,} | {b['p1']:.1f}% | {b['p2']:.1f}% | {b['cb']:.1f}% |")
w("")
w("## Setup cells that differ from baseline (ES, q < 0.05)")
w("")
w("| Timeframe | C2 | C1 (setup) | Direction | Trades | +1R before stop | vs baseline (points) | q |")
w("|---|---|---|---|---|---|---|---|")
for c in sig:
    w(f"| {c['tf']} | {c['c2']} | {c['c1']} | {c['dir']} | {int(c['n']):,} | {c['p1']:.1f}% | {c['d1']:+.1f} | {max(c['q1'], 0.001):.3f} |")
w("")
w("Trades that follow one another share price paths, so q-values are optimistic. Treat these as reasons to look again, not as proof.")
w("")
w("## Limitations")
w("")
w("- One trade definition: a symmetric 1R bracket over three bars. Exits, targets (magnitude, exhaustion) and management are not tested.")
w("- No commissions or slippage. In-sample description; a dev (to 2022) and holdout (2023 on) split is on the interactive page.")
w("- Unfillable data gaps: 2019-02-26 19:40-22:45 ET and 2020-06-30 18:00-19:59 ET. Nothing is interpolated.")
w("")
w("## Reproduce")
w("")
w("Everything on the page is rebuilt from 1-minute bars by open code: https://github.com/345-strategies/strat-enrichment (bar types in strat_enrichment/, the statistics in stats/). "
  "Market data is licensed and not redistributed, so bring your own.")
w("")
w("## Credit")
w("")
w("Format and trade definition after The Strat Stats (NQ) by @notprofgreen: https://x.com/notprofgreen. TheStrat methodology by Rob Smith. "
  "Bar-type colours and definitions follow TheStrat Suite: https://thestratsuite.com")
open(dst, "w").write("\n".join(L) + "\n")
print(dst, len(L))
