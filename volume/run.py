"""Run the whole volume study on your own 1-minute files.

    python volume/run.py DATA_DIR OUT_DIR

DATA_DIR holds one input file per instrument, named by what the analysis expects. Any you leave out are
skipped, and the tables that need them say so.

    nq.parquet, es.parquet                      futures, 2018 (ES 2020) to 2026, the main sample
    spy.parquet                                 stocks, 2018-05 to 2026-04
    spyfwd.parquet, qqqfwd.parquet              stocks, the forward window (2026-06-08 to 2026-10-19 in the report)
    iwmfwd, aaplfwd, nvdafwd, tslafwd .parquet  stocks, round 4 confirmation symbols (same forward window)
    eslate.parquet                              futures, ES 2026-05-15 onward (round 4 confirmation)

Writes OUT_DIR/work/ (trade tables) and OUT_DIR/RESULTS*.md, one per round, the same tables as the report."""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FUTURES = {"nq", "es", "eslate"}
ORDER = ["nq", "es", "spy", "spyfwd", "qqqfwd", "iwmfwd", "aaplfwd", "nvdafwd", "tslafwd", "eslate"]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data, out = sys.argv[1], sys.argv[2]
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)
    have = [i for i in ORDER if os.path.exists(os.path.join(data, f"{i}.parquet"))]
    if not {"nq", "es", "spy"} & set(have):
        raise SystemExit(f"no nq/es/spy input in {data}")
    for i in have:
        mk = "futures" if i in FUTURES else "stocks"
        subprocess.run([sys.executable, f"{HERE}/volstudy.py", os.path.join(data, f"{i}.parquet"), i, work, "--market", mk], check=True)
    main3 = [i for i in ("nq", "es", "spy") if i in have]
    steps = [
        ("analyze.py", "RESULTS.md", main3),                                         # round 1: H1-H9, R1-R2
        ("analyze2.py", "RESULTS_ROUND2.md", main3 + (["spyfwd"] if "spyfwd" in have else [])),  # round 2: E1-E4
        ("analyze3.py", "RESULTS_ROUND3.md", []),                                   # round 3: forward check of 4 leads
        ("analyze4.py", "RESULTS_ROUND4.md", []),                                   # round 4: C, M, Z
        ("lit.py", "RESULTS_ROUND5.md", []),                                        # round 5: literature
    ]
    for script, name, args in steps:
        r = subprocess.run([sys.executable, f"{HERE}/{script}", work, os.path.join(out, name), *args],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print(f"{name}: {'ok' if r.returncode == 0 else 'failed: ' + r.stderr.strip().splitlines()[-1]}", flush=True)


if __name__ == "__main__":
    main()
