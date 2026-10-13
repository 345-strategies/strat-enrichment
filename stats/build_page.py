"""Write a Strat Stats page from an atlas JSON.

Usage: python stats/build_page.py atlas.json page.html --title "My Strat Stats"
  --nq FILE, --nq-replication FILE    the NQ comparison data; without them the vs NQ tabs are hidden
  --repo URL                 the link on the Reproduce tab
"""

import argparse
import html
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")
REPO = "https://github.com/345-strategies/strat-enrichment"


def embed(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def main() -> None:
    p = argparse.ArgumentParser(description="Write a Strat Stats page from an atlas JSON.")
    p.add_argument("atlas", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--title", required=True)
    p.add_argument("--nq", type=Path)
    p.add_argument("--nq-replication", type=Path)
    p.add_argument("--repo", default=REPO)
    a = p.parse_args()

    rep = json.loads(a.nq_replication.read_text()) if a.nq_replication else None
    if rep:
        rep.pop("cells", None)
    page = (
        TEMPLATE.read_text()
        .replace("__REPO__", json.dumps(a.repo))
        .replace("__NQREP__", embed(rep))
        .replace("__NQ__", embed(json.loads(a.nq.read_text()) if a.nq else None))
        .replace("__TITLE__", html.escape(a.title))
        .replace("__DATA__", embed(json.loads(a.atlas.read_text())))
    )
    a.out.write_text(page)
    print(a.out, len(page))


if __name__ == "__main__":
    main()
