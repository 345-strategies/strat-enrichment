#!/bin/sh
# Build the 345strategies.com Strat Stats files (strat-stats.html, .md, .json) from atlas outputs.
# Usage: stats/site/build.sh <atlas_es.json> <nq_compare.json> <nq_replication.json> <outdir> <updated YYYY-MM-DD>
# nq_compare.json is @notprofgreen's NQ page data (prep_nq_compare.py on his saved page). nq_replication.json is
# atlas.py run on NQ over his window and timeframes (--timeframes 15m,30m,1h,2h,4h,6h,8h,D,W), then compare_nq.py.
set -eu
here=$(dirname "$0")
es=$1 nq=$2 rep=$3 out=$4 updated=$5
mkdir -p "$out"
python "$here/../build_page.py" "$es" "$out/page.html" --title "Strat Stats" --nq "$nq" --nq-replication "$rep"
python "$here/make_summary_md.py" "$es" "$rep" "$out/strat-stats.md"
python "$here/make_site_page.py" "$out/page.html" "$out/strat-stats.md" "$es" "$out/strat-stats.html" "$updated"
python "$here/make_data_json.py" "$es" "$rep" "$out/strat-stats.json" "$updated"
node "$here/prerender.js" "$out/strat-stats.html"
rm "$out/page.html"
