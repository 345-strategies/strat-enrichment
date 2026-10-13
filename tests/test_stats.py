"""The Strat Stats atlas and page builder, run on the sample."""

import json
import subprocess
import sys
from pathlib import Path

from conftest import SAMPLE

STATS = Path(__file__).resolve().parent.parent / "stats"


def test_atlas_and_page_on_the_sample(tmp_path):
    atlas = tmp_path / "atlas.json"
    subprocess.run([sys.executable, STATS / "atlas.py", "futures", SAMPLE, atlas, "--name", "SYN"], check=True, capture_output=True)
    d = json.loads(atlas.read_text())
    assert d["info"]["market"] == "SYN" and d["info"]["session"] == "futures"
    assert d["info"]["trade_tfs"] == ["15m", "30m", "1h", "4h", "D"]
    assert {r["type"] for r in d["freq"]} == {"1", "2U", "2D", "3"}
    assert {r["c1"] for r in d["setups"]} <= {"1", "2U", "F2U", "2D", "F2D", "3U", "3D"}

    page = tmp_path / "page.html"
    subprocess.run([sys.executable, STATS / "build_page.py", atlas, page, "--title", "SYN Strat Stats"], check=True, capture_output=True)
    html = page.read_text()
    assert "<title>SYN Strat Stats</title>" in html
    assert "__" not in html.split("<script>")[1].split("const TF")[0]  # every placeholder filled
    assert "const NQ = null;" in html
