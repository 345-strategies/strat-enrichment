# Strat Enrichment

Rebuild TheStrat bar states on every timeframe from 1-minute bars.

I use this to label every bar behind the [Strat Stats](https://345strategies.com/research/strat-stats) page. It is public so you can rebuild the same states on your own data and check my numbers, or build your own research on top of them.

Give it 1-minute bars. It builds session-anchored 15m, 30m, 1h, 4h, D, W and M bars, and labels each completed bar with its type, its sign, whether it took the prior bar's high or low, its 2-bar and 3-bar combos, and timeframe continuity. It also writes every timeframe's live, forming state at the close of every 1-minute bar.

Every definition comes from the [TheStrat grammar](https://github.com/345-strategies/thestrat-suite/tree/main/grammar), the same package that is held to TheStrat Suite's Pine code by parity tests. This repo builds the bars and decides which candle is compared with which. It does not define anything on its own.

## Try it on the sample

```bash
git clone https://github.com/345-strategies/strat-enrichment
cd strat-enrichment
pip install -e .
strat-enrich sample/futures_1m.parquet out/
```

`sample/futures_1m.parquet` is synthetic: a random walk on CME futures hours, with a contract roll in March. It is not market data. Rebuild it with `python -m strat_enrichment.sample sample/futures_1m.parquet`.

Python 3.10 or newer. The install pulls numpy, pandas, pyarrow and the grammar.

## Run it on your own data

Input is a parquet file of 1-minute bars:

| column | meaning |
|---|---|
| `ts` | bar start, UTC epoch seconds |
| `o`, `h`, `l`, `c` | prices |
| `contract` | futures only: the contract each bar comes from |

For futures, give raw contract prices plus `contract`. At each change of contract the gap between the new contract's first open and the old one's last close is removed from every earlier bar, so the latest contract stays at real prices and a roll can't print a fake 2 or 3. That gap is measured between the two minutes either side of the switch, so switch contracts in the middle of a session. A switch across a weekend would also remove the weekend's move.

```bash
strat-enrich minutes.parquet out/ --market futures
strat-enrich minutes.parquet out/ --market stocks --tfc htf=D,W,M --tfc day=15m,1h,4h,D
```

| option | default | |
|---|---|---|
| `--market` | `futures` | `futures`: sessions open 18:00 ET and are dated by the day they close. `stocks`: 09:30 to 16:00 ET only; pre- and post-market minutes are dropped. |
| `--timeframes` | `15m,30m,1h,4h,D,W,M` | any `Nm` or `Nh`, plus `D`, `W`, `M` |
| `--tfc NAME=TF,...` | `all` = every timeframe | a named set of timeframes to read continuity across; repeat for more sets |

On 8.7 years of ES (3.1 million minutes) a run takes about a minute and 2.5 GB of memory.

From Python:

```python
import pandas as pd
from strat_enrichment import enrich

out = enrich(pd.read_parquet("minutes.parquet"), market="futures", tfc_sets={"htf": ["D", "W", "M"]})
out.bars      # completed bars, every timeframe
out.minutes   # every minute with each timeframe's forming state
out.rolls     # contract switches and the gap removed at each
```

## Sessions and timeframes

Every timeframe counts from the session open, not from midnight.

- **Futures**: the session opens at 18:00 ET. Intraday bars are anchored there, so 4h bars start at 18:00, 22:00, 02:00, 06:00, 10:00 and 14:00 ET. D is the session, so Sunday 18:00 opens Monday's bar.
- **Stocks**: 1h bars start at 09:30, 10:30 and so on, and the last one is 30 minutes. 4h bars start at 09:30 and 13:30.
- **W** is the ISO week of the session date and **M** its calendar month.

Times are ET wall clock, so the anchor follows daylight saving time. Both markets are closed when the clocks change. If your data trades through a change, the run stops with an error instead of merging two real hours into one bar.

## Output

`out/bars.parquet`: one row per completed bar, every timeframe stacked. The last bar of each timeframe is left out because the file can end before it closes.

| column | |
|---|---|
| `tf` | `15m` ... `M` |
| `session` | session date |
| `ts`, `ts_last` | first and last 1-minute bar in it (UTC epoch seconds) |
| `o`, `h`, `l`, `c`, `minutes` | OHLC and how many 1-minute bars it holds |
| `type` | `1u` `1d` `2u` `2d` `F2u` `F2d` `3u` `3d`, chart notation from the grammar; empty on the first bar |
| `above_open` | closed above its own open |
| `took_c1h`, `took_c1l` | traded above the prior bar's high, below its low |
| `combo2`, `combo3` | e.g. `2d-2u`, `2d-1-2u`, oldest first |
| `tfc_<name>` | `up`, `down` or `conflict` across that set at the bar's close |

`out/minutes.parquet`: one row per 1-minute bar.

| column | |
|---|---|
| `ts`, `session`, `o`, `h`, `l`, `c` | the minute, roll-adjusted |
| `contract`, `adj` | futures: the contract, and the offset added (raw price = price - adj) |
| `<tf>_type`, `<tf>_above_open`, `<tf>_took_c1h`, `<tf>_took_c1l` | that timeframe's forming bar so far (its open, running high and low, this minute's close) against its last completed bar |
| `tfc_<name>` | continuity across that set at this minute |

`out/rolls.json`: each contract switch and the gap removed.

At each bar's last minute the forming state has to equal the completed bar's type. Every run checks that on every bar and stops if it doesn't hold.

## Definitions

These come from the grammar's [SPEC.md](https://github.com/345-strategies/thestrat-suite/blob/main/grammar/SPEC.md). The ones that usually cause disagreements:

- **Equal is not a break.** A high that only matches the prior high has not taken it.
- **2u and 2d** name the side that broke, not the candle's color. A 2u can close red.
- **3** broke both sides. It is 3u if it closed above its own open, 3d otherwise. A 3 never fails.
- **Failed 2** (F2u, F2d): a 2 that closed back inside the prior bar's range, edges included. Only 2s fail.
- **Above open** means the close is above the open. A close at the open is not above.
- **Continuity** reads only the sign channel: `up` when every timeframe in the set is above its open, `down` when none is.
- **Combos** read oldest to newest. An inside bar in the middle of a combo is written bare (`2d-1-2u`).

## Strat Stats

`stats/` has the code behind the [Strat Stats](https://345strategies.com/research/strat-stats) page. It measures every first break of a closed bar on 1-minute data against a baseline. Bars and types come from this package.

```bash
python stats/atlas.py futures minutes.parquet atlas.json --name ES
python stats/build_page.py atlas.json page.html --title "My Strat Stats"
```

Use `futures` or `stocks`. `--tick` sets the stop offset (default 0.25 and 0.01) and `--timeframes` the bars measured. The 345strategies.com page is this template plus `stats/site/head.html` and `stats/site/nav.html` (`--head`, `--nav`) and the NQ comparison data (`--nq`, `--nq-replication`).

## What this does not do

No signals, entries, targets, stops, levels or backtests. It labels bars. What you do with the labels is up to you.

My data is licensed and can't be shared, so bring your own 1-minute bars.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

They check every completed bar and a random sample of forming states against the grammar, recomputed from scratch, plus session anchors across DST, roll removal, stock hours, each definition above, and the commands in this README.

## License

Mozilla Public License 2.0, same as TheStrat Suite.
