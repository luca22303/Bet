# Bet — Bundesliga analytics

A point-in-time data spine and forecast backtesting harness for Bundesliga
match prediction, built on free public data.

This is layer one of a larger plan (xG and event data, an LLM news/injury layer,
a Dixon-Coles forecasting engine, odds ingestion and +EV alerting). It is
deliberately the unglamorous layer, because every layer above it is worthless
without it.

## Why this layer first

Two failure modes destroy sports betting models, and both are invisible in the
output:

**Lookahead leakage.** Train on matchday 20 using season totals that already
include matchday 20 and your backtest prints a profit that does not exist. The
store makes this structurally impossible: every fact carries `known_at`, and
models read the world only through `WHERE known_at <= as_of`. `bet check`
verifies it, and the test suite injects a deliberate leak to confirm the guard
actually fires.

**No benchmark.** A model that produces probabilities will always find "value"
somewhere. The only question that matters is whether it beats the devigged
closing line, and you cannot answer that without historical closing odds and a
walk-forward harness. Both are here.

## What the demo shows

Running the harness on synthetic seasons — where a simulated bookmaker prices
off the *true* probabilities, so no edge exists by construction:

```
     model    n     rps  log_loss   brier  accuracy     ece  rps_skill_vs_market
    market 1341 0.19554   0.97243 0.57950   0.53691 0.01674              0.00000
       elo 1341 0.20311   0.99731 0.59559   0.51305 0.02667             -0.03871
home_prior 1341 0.23155   1.07787 0.65269   0.42506 0.00447             -0.18412

Signals the elo model would fire against a book that knows the truth:
  ignoring tax : 1153
  with 5.3% tax:  872
```

The Elo model is measurably *worse* than the market, and still fires 1,153
confident "value" bets. Every one is a false positive. This is what an
uncalibrated model does with real money, and it is why the evaluation layer
exists before the alerting layer.

## Install

```bash
pip install -e ".[dev]"
```

## Use

```bash
bet init                                  # create the schema
bet ingest --source football_data --seasons 2010-2026
bet ingest --source all --seasons 2015-2026
bet check                                 # point-in-time integrity
bet backtest --from 2018-08-01 --calibration
```

Ingest `football_data` first — it carries results *and* historical closing odds
from several books including Pinnacle, which is what makes closing-line-value
measurement possible at zero cost.

## Data sources

All free, no API keys.

| Source | Provides | Notes |
|---|---|---|
| football-data.co.uk | results + opening/closing odds, back to the 1990s | the backbone; without it there is no CLV measurement |
| ClubElo | cross-division power ratings | supplies the prior for promoted sides, which results-only models get badly wrong |
| Understat | shot-level xG | converges far faster than goals on a 306-match season |
| OpenLigaDB | fixtures, results, matchday structure | a plain public API, no scraping grey area |

Scraped sources are rate-limited to one request every three seconds and every
payload is archived to `data/raw/` before parsing, so a site changing its markup
costs a re-parse rather than a re-scrape.

## Layout

```
src/bet/
  config.py          paths, constants, the German stake tax rate
  teams.py           canonical club ids; cross-source name resolution
  store/             DuckDB schema and point-in-time reads
  ingest/            source adapters (archive first, then parse)
  odds/devig.py      multiplicative / additive / power / Shin
  models/            Model contract + the three baselines to beat
  evaluation/        RPS, log loss, Brier, calibration, walk-forward, CLV
  ev.py              expected value and Kelly, with tax made explicit
```

## Three things worth knowing

**Devigging method changes the answer.** On a heavy favourite, multiplicative
and power devigging disagree by ~10% in relative terms on the longshot — larger
than any edge you are likely to find. The default is Shin. Never trust a value
signal that does not survive a method change.

**The German stake tax is not a rounding error.** The
Rennwett- und Lotteriegesetz levies 5.3% on sports betting stakes. An even-money
bet needs 52.8% rather than 50% to break even, and a claimed +10% EV becomes
+4.2%. `bet.ev` requires a tax mode rather than defaulting to none. Confirm how
your operator actually applies it before trusting any number.

**Measure CLV, not ROI.** A Bundesliga season is ~300 bets. At a true 2% edge
the standard error on seasonal ROI exceeds the edge, so a losing season and a
winning season are both consistent with having an edge and with not having one.
Closing line value converges in weeks instead of years.

## Not yet built

- Dixon-Coles forecasting engine (time-weighted MLE, with identifiability
  constraints on both attack and defence, and bounded rho)
- xG-based team strength estimation
- LLM injury/news extraction with schema validation
- Live odds ingestion and +EV alerting

## Testing

```bash
make test
```

73 tests, no network required. Synthetic seasons are generated from known team
strengths, so models are checked for recovering the truth rather than merely for
running without raising.
