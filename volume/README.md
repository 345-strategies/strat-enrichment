# Volume study

Does bar volume help TheStrat signals? This folder is the code behind the
[volume study](https://345strategies.com/research/volume) page: six pre-registered rounds on NQ, ES and SPY,
2018-2026, plus a seventh on Bitcoin. Every fill, stop and exit is checked on 1-minute bars.

The short answer was no. Volume did not predict the direction of a break, did not improve exits or stops, and
its small gain in forecasting how far price travels changed no decision we tested. The page has the numbers and
the caveats. This code lets you check them on your own data.

## What you need

- Python 3.10 or newer, and this repo installed: `pip install -e ".[dev]"`. Add `pip install databento` if you
  pull bars from Databento.
- 1-minute bars with volume. Our data is licensed and can't be shared. Futures and SPY came from vendor feeds
  (Databento has all of it for a few dollars); Bitcoin comes free from Binance's public archive.

Check the install on synthetic bars first. This runs the whole pipeline in under a minute:

```bash
pytest tests/test_volume.py
```

## 1. Prepare the inputs

Every input is a parquet file of 1-minute bars: `ts` (bar start, UTC epoch seconds), `o`, `h`, `l`, `c`, `v`,
plus `contract` for futures. Name each file the way `run.py` expects (`nq.parquet`, `es.parquet`, ...).

**Futures** (CME via Databento, every contract of the product):

```python
import databento as db
c = db.Historical()   # reads DATABENTO_API_KEY
c.timeseries.get_range(dataset="GLBX.MDP3", schema="ohlcv-1m", stype_in="parent",
    symbols=["NQ.FUT"], start="2018-01-01", end="2026-09-23").to_file("nq_raw.dbn.zst")
c.timeseries.get_range(dataset="GLBX.MDP3", schema="ohlcv-1m", stype_in="parent",
    symbols=["ES.FUT"], start="2019-12-01", end="2026-05-16").to_file("es_raw.dbn.zst")
```

```bash
python volume/prep_futures.py NQ data/nq.parquet nq_raw.dbn.zst --end 2026-09-22
python volume/prep_futures.py ES data/es.parquet es_raw.dbn.zst --start 2020-01-01 --end 2026-05-15
```

`prep_futures.py` keeps one quarterly contract per minute and switches at 00:00 ET on the Tuesday of expiration
week. That is the roll in our data; on every minute where both exist, its output matches our NQ file exactly
(contract, close and volume). The engine drops the three sessions before each switch, because volume has left
the expiring contract by then.

**SPY and other stocks** (any 1-minute source with volume):

```bash
python volume/prep_stocks.py data/spy.parquet spy_bars.dbn.zst --start 2018-05-01 --end 2026-04-21
python volume/prep_stocks.py data/spyfwd.parquet spy_fwd.csv --start 2026-05-08 --end 2026-10-19
python volume/prep_stocks.py data/qqqfwd.parquet qqq_fwd.csv --start 2026-05-08 --end 2026-10-19
```

Our SPY used consolidated prices with NYSE Arca volume (Databento `ARCX.PILLAR`). To copy that, give the
consolidated bars as the file and the Arca bars with `--volume-from`. The forward files start 20 sessions before
the first test day (2026-06-08), because relative volume needs 20 prior sessions. Round 4 also reads
`iwmfwd`, `aaplfwd`, `nvdafwd`, `tslafwd` (same window) and `eslate` (ES from 2026-05-15).

**Bitcoin** (free, from data.binance.vision):

```bash
python volume/fetch_binance.py BTCUSDT 2020-01 2026-05 raw/binance/
python volume/prep_crypto.py data/btc.parquet raw/binance/*.zip --start 2020-01-01 --end 2026-05-04
```

Our second venue was Coinbase BTC-USD from 2018 (`btccb.parquet`, from Coinbase's public candles API, any
1-minute export with time, open, high, low, close and volume works with `prep_crypto.py`).

**Single stocks** (Databento `XNAS.ITCH`, Nasdaq-exchange volume only, about $5 for five stocks 2018-2026):

```bash
python volume/prep_stocks.py data/aapl.parquet aapl.dbn.zst --start 2018-05-01 --end 2026-10-23 --splits 2020-08-31:4
python volume/prep_stocks.py data/msft.parquet msft.dbn.zst --start 2018-05-01 --end 2026-10-23
python volume/prep_stocks.py data/nvda.parquet nvda.dbn.zst --start 2018-05-01 --end 2026-10-23 --splits 2021-07-20:4,2024-06-10:10
python volume/prep_stocks.py data/amzn.parquet amzn.dbn.zst --start 2018-05-01 --end 2026-10-23 --splits 2022-06-06:20
python volume/prep_stocks.py data/tsla.parquet tsla.dbn.zst --start 2018-05-01 --end 2026-10-23 --splits 2020-08-31:5,2022-08-25:3
```

`--splits DATE:ratio,...` divides prices before each split date by the ratio and multiplies volume by it.
Round 9 used META, AMD, GOOGL (`2022-07-18:20`), AVGO (`2024-07-15:10`) and COST the same way.

## 2. Run

Rounds 1-5 (NQ, ES, SPY and the forward symbols):

```bash
python volume/run.py data/ results/
```

writes `results/RESULTS.md` (round 1: direction H1-H9, range R1-R2), `RESULTS_ROUND2.md` (E1-E4),
`RESULTS_ROUND3.md` (forward check of four leads), `RESULTS_ROUND4.md` (confirmation C, management M, sizing Z)
and `RESULTS_ROUND5.md` (published claims). Missing instruments are skipped.

Round 7, Bitcoin:

```bash
python volume/volstudy.py data/btc.parquet btc results/work --market crypto
python volume/volstudy.py data/btccb.parquet btccb results/work --market crypto
cd volume && python analyze_btc.py ../results/work ../results/RESULTS_ROUND7_BTC.md
python lit.py ../results/work ../results/RESULTS_ROUND7_BTC_LIT.md btc btccb
```

For crypto the session is the UTC day, all 1,440 minutes; costs are 0.10% of price round trip.

Round 8, single stocks (0.02% round trip; the analysis reports every test after and before that cost):

```bash
for s in aapl msft nvda amzn tsla; do
  python volume/volstudy.py data/$s.parquet $s results/work --market stocks --cost-pct 0.0002
done
cd volume && python analyze_stocks.py ../results/work ../results/RESULTS_ROUND8_STOCKS.md aapl msft nvda amzn tsla
```

`analyze_stocks.py` runs the round-1 family on each stock and on all of them pooled (one cluster per date).

## How close you should get

Fed the same bars, this code reproduces our published tables exactly for NQ and ES. For SPY, the published
continuity rows (H6, the TFC-by-trigger-volume contrast, and lead L2) came from an earlier version with a bug: the
60-minute open for trades triggered in SPY's last half hour (15:30-16:00 ET) was never set, so those trades'
continuity flag was effectively random. This version fixes it and the page quotes the corrected numbers; no
verdict changed.

With bars pulled fresh from a vendor, expect small differences in the last digits (vendor corrections, a missing
minute, a different SPY source). The verdicts should not move. If one does, please open an issue.

Not reproducible from here: round 6 (sizing a breakout system and timing SPY 0DTE straddle sales), because that
system's trade lists and the option quotes are not public.

## Files

| file | what it does |
|---|---|
| `volstudy.py` | the engine: builds each session's 1-minute grid, volume baselines, the trades and their outcomes |
| `analyze.py` | round 1, the pre-registered direction family H1-H9 and range R1-R2 |
| `analyze2.py` | round 2: range with strong price controls, a range-based exit rule, signed volume, a big-day call |
| `analyze3.py` | round 3: forward check of four leads |
| `analyze4.py` | round 4: confirmation on new symbols, management on 1m, day-range sizing |
| `lit.py` | round 5: published claims about volume |
| `analyze_btc.py` | round 7: the same tests on Bitcoin |
| `analyze_stocks.py` | rounds 8-9: the same tests on single stocks, pooled and one by one, after and before costs |
| `run.py` | runs rounds 1-5 end to end |
| `prep_futures.py`, `prep_stocks.py`, `prep_crypto.py`, `fetch_binance.py` | turn vendor bars into inputs |

Research only. Nothing here is a trading rule.
