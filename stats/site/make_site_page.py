"""Wrap the built Strat Stats page for 345strategies.com: head tags, JSON-LD, a static summary and links
for crawlers, answer engines and agents. Prerender the default tab afterwards with prerender.js.

Usage: python make_site_page.py <page.html from build_page.py> <strat-stats.md> <atlas.json> <out.html> <updated YYYY-MM-DD>
"""
import html, json, re, sys

src, md_path, atlas_path, dst, updated = sys.argv[1:6]
page = open(src).read()
md = open(md_path).read()
info = json.load(open(atlas_path))["info"]
URL = "https://345strategies.com/research/strat-stats"
TITLE = "Strat Stats: ES and NQ futures, 2018-2026"
DESC = ("Every TheStrat combo's first break on ES and NQ futures, 2018-2026, measured on 3.07M one-minute bars "
        "against a baseline, with failed 2s split out and a replication of @notprofgreen's NQ study. Research-only.")
LEDE = ("Strat Stats measures the first break of every TheStrat combo on E-mini S&amp;P 500 (ES) futures, "
        f"{info['start']} to {info['end']}, from {info['minutes']:,} one-minute bars, against a baseline of all breaks. "
        "Entries are on the bar after the pattern completes; failed 2s are split out. The headline: the combo label alone "
        "barely beats a random break, on ES and on NQ. Descriptive statistics, not a trading system.")


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(https://[^\s)|]+?)([.,;]?)(?=[\s)]|$)", r'<a href="\1">\1</a>\2', t)


def md_to_html(text):
    """Enough Markdown for the summary: headings, paragraphs, bullet lists, pipe tables."""
    out, lines, i = [], text.splitlines(), 0
    ids = {"Definitions": "definitions", "Key findings": "findings", "Limitations": "limitations",
           "Reproduce": "reproduce", "Credit": "credit"}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            h = ln[3:]
            hid = ids.get(h) or re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-")
            out.append(f'<h2 id="{hid}">{inline(h)}</h2>')
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                it = lines[i][2:]
                iid = ' id="entry-bar"' if it.startswith("Entry bar") else ' id="failed-2"' if it.startswith("Failed 2") else ""
                items.append(f"<li{iid}>{inline(it)}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-| ]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f'<div class="tbl"><table class="txt"><tr>{head}</tr>{body}</table></div>')
            continue
        elif ln.strip() and not ln.startswith("# "):
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


body_md = md.split("## Key findings", 1)[1]
summary = md_to_html("## Key findings" + body_md)

ld = {"@context": "https://schema.org", "@graph": [
    {"@type": "Dataset", "@id": URL + "#dataset", "name": TITLE, "description": DESC, "url": URL,
     "temporalCoverage": f"{info['start']}/{info['end']}", "isAccessibleForFree": True,
     "keywords": ["TheStrat", "Strat combos", "failed 2", "ES futures", "NQ futures", "price action statistics", "backtest"],
     "variableMeasured": ["share of first breaks reaching +1R before the stop", "share reaching +2R before the stop",
                          "trigger bar close beyond entry", "lift versus baseline (percentage points)", "Benjamini-Hochberg q-value"],
     "measurementTechnique": "First break of the setup bar sequenced on 1-minute bars; stop at the setup bar's other side plus one tick; 3-bar outcome window",
     "creator": {"@id": "https://345strategies.com/#org"},
     "isBasedOn": {"@type": "CreativeWork", "name": "The Strat Stats (NQ)", "author": {"@type": "Person", "name": "@notprofgreen", "url": "https://x.com/notprofgreen"}},
     "distribution": [
         {"@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": "https://345strategies.com/research/strat-stats.json"},
         {"@type": "DataDownload", "encodingFormat": "text/markdown", "contentUrl": "https://345strategies.com/research/strat-stats.md"}]},
    {"@type": "Article", "headline": TITLE, "description": DESC, "url": URL, "datePublished": "2026-10-13", "dateModified": updated,
     "author": {"@id": "https://345strategies.com/#org"},
     "publisher": {"@id": "https://345strategies.com/#org"}, "about": {"@id": URL + "#dataset"},
     "image": "https://345strategies.com/assets/strat-stats-og.png"},
    {"@type": "Organization", "@id": "https://345strategies.com/#org", "name": "345 Strategies", "url": "https://345strategies.com/"}]}

head = f'''<title>{TITLE} | 345 Strategies</title>
<meta name="description" content="{html.escape(DESC)}">
<link rel="canonical" href="{URL}">
<link rel="alternate" type="text/markdown" href="/research/strat-stats.md" title="Plain-text summary">
<link rel="alternate" type="application/json" href="/research/strat-stats.json" title="Results data">
<meta property="og:type" content="article">
<meta property="og:site_name" content="345 Strategies">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{html.escape(DESC)}">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="https://345strategies.com/assets/strat-stats-og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Strat Stats: the combo label alone barely beats a random break on ES and NQ futures">
<meta property="article:published_time" content="2026-10-13">
<meta property="article:modified_time" content="{updated}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{html.escape(DESC)}">
<meta name="twitter:image" content="https://345strategies.com/assets/strat-stats-og.png">
<meta name="twitter:creator" content="@spintrades">
<script type="application/ld+json">{json.dumps(ld, separators=(",", ":"))}</script>'''
page, n = re.subn(r"<title>[^<]*</title>", lambda _: head, page, count=1)
assert n == 1, "no <title> in the built page"
page = page.replace("</style>", """.lede-top{max-width:90ch;font-size:15px;line-height:1.6;margin:4px 0 6px}
.meta-top{color:var(--dim);font-size:13px;margin:0 0 14px}.meta-top a{color:var(--dim)}
#about{border-top:1px solid var(--line);margin-top:36px;padding-top:8px;max-width:110ch}
#about h2{font-size:18px;margin:22px 0 8px}#about li{margin:4px 0}
</style>""", 1)
page = page.replace('<div class="wrap">\n  <header>',
    '<div class="wrap">\n  <a href="/" style="display:inline-block;margin:0 0 14px;color:var(--dim);font:500 13px/1 \'IBM Plex Mono\',monospace;text-decoration:none;letter-spacing:.08em">&larr; 345 STRATEGIES</a>\n  <header>', 1)
page = page.replace('  <div class="types" id="types"></div>',
    f'  <p class="lede-top">{LEDE}</p>\n  <p class="meta-top">Updated <time datetime="{updated}">{updated}</time> · Past statistics, not advice; no costs or slippage · '
    '<a href="#definitions">Definitions</a> · <a href="/research/strat-stats.json">Data (JSON)</a> · <a href="/research/strat-stats.md">Plain text</a></p>\n'
    '  <div class="types" id="types"></div>', 1)
page = page.replace('  <main id="view"></main>\n</div>',
    f'  <main id="view"></main>\n  <section id="about" aria-label="Summary and definitions">\n{summary}\n  </section>\n</div>', 1)
for tag in ('id="about"', 'class="lede-top"', '345 STRATEGIES', 'application/ld+json'):
    assert tag in page, tag
open(dst, "w").write(page)
print(dst, len(page))
