# Bet — Bundesliga analytics

A point-in-time data spine, a forecast backtesting harness, and a corrected
Dixon-Coles engine for Bundesliga match prediction, built on free public data.

The harness came first on purpose. Every layer above it — the forecasting
engine, and later the news/injury layer and +EV alerting — is worthless without
a way to tell whether it works.

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

Running on synthetic seasons — where a simulated bookmaker prices off the *true*
probabilities, so no edge exists by construction:

```
      model    n     rps  log_loss   brier  accuracy     ece  rps_skill_vs_market
     market 1341 0.19554   0.97243 0.57950   0.53691 0.01674              0.00000
dc_no_decay 1341 0.19886   0.98417 0.58654   0.53468 0.01107             -0.01699
   dc_decay 1341 0.20045   0.98956 0.59000   0.52796 0.01447             -0.02510
        elo 1341 0.20311   0.99731 0.59559   0.51305 0.02667             -0.03871
 home_prior 1341 0.23155   1.07787 0.65269   0.42506 0.00447             -0.18412

Signals the elo model would fire against a book that knows the truth:
  ignoring tax : 1153
  with 5.3% tax:  872
```

Two things worth reading carefully.

Dixon-Coles beats Elo and closes most of the gap to the market, but it does not
beat the market — and on synthetic data where the book is *exactly* right, it
never should. On real data that gap is the whole question.

The Elo model, meanwhile, is measurably worse than the market and still fires
1,153 confident "value" bets. Every one is a false positive. That is what an
uncalibrated model does with real money, and it is why the evaluation layer was
built before the alerting layer.

(`dc_no_decay` beating `dc_decay` is a property of the synthetic data, where
team strengths never change, so decay only discards information. On real data
squads turn over and the ranking reverses — which is precisely why `bet tune`
exists rather than a hardcoded constant.)

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
bet tune --from 2018-08-01                # choose the decay rate empirically
bet predict --days 10                     # price the next fixtures
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
  models/            Model contract, baselines, Dixon-Coles, promoted-team prior
  evaluation/        RPS, log loss, Brier, calibration, walk-forward, CLV
  ev.py              expected value and Kelly, with tax made explicit
```

## The forecasting engine

Time-weighted Dixon-Coles, with four corrections the original paper leaves
implicit:

**Identifiability.** The rates are `exp(c + attack_home + defence_away + home)`.
Constraining only the attack vector to sum to zero is not enough — adding `k` to
every attack and subtracting `k` from every defence leaves both rates unchanged,
so the likelihood has a flat ridge and the optimiser returns different
parameters on every refit. Both vectors are constrained, with an explicit
intercept carrying the level. Measured refit drift on shuffled input: `4e-07`.

**Bounded rho.** The low-score correction can drive joint probabilities negative
— the (0,0) cell is `1 - lambda*mu*rho` — so an unconstrained fit returns
"probabilities" that are not probabilities. The likelihood rejects violating
parameters, and prediction clips rho to each fixture's own valid range.

**Decay tuned, not assumed.** `xi` is a real parameter: too slow and the model
believes in a squad that has since been sold, too fast and it fits three weeks
of noise. `bet tune` selects it by out-of-sample log loss.

**Promoted teams.** A results-only model gives a newly promoted side the league
average, and stays wrong into October — on exactly the fixtures where it will
think it has found the most value. `PromotedTeamPrior` learns the mapping from
ClubElo rating to fitted strength *from the established teams*, then reads off
the promoted side's rating. The blend ramps with matches played, so there is no
discontinuity mid-season. No hand-tuned constants.

Targets: `goals` (classical), `xg` (strengths from expected goals, which
converge far faster on a 306-match season), or `blend` (geometric blend of the
two rate sets). Rho always comes from the goals fit, since xG has no notion of a
1-1.

Validation on synthetic seasons generated from known strengths: home advantage
recovered as 0.2503 against a true 0.26, attack correlation 0.969, defence 0.980.

All markets derive from a single scoreline matrix, so the 1X2 price and the
totals price can never imply different scorelines.

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

- LLM injury/news extraction with schema validation
- Live odds ingestion and +EV alerting
- Player-level prop modelling (where the softer markets are)
- Bayesian hierarchical variant, for parameter uncertainty that Kelly can use

## Testing

```bash
make test
```

113 tests, no network required. Synthetic seasons are generated from known team
strengths, so models are checked for recovering the truth rather than merely for
running without raising.
