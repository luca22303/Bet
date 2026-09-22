# Bet — Bundesliga analytics

A Bundesliga analytics platform built on free public data: a point-in-time data
spine, a walk-forward backtesting harness, a corrected Dixon-Coles match engine,
a player prop model, spatial shot analysis, LLM-extracted team news, and a
matchday brief that pulls it together.

The harness came first on purpose. Every layer above it is worthless without a
way to tell whether it works.

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
bet props --stat shots --days 7           # player prop prices
bet scout --team bayern_munich            # shot profile and spatial summary
bet news --file news.txt --team bayern_munich   # extract availability (local LLM)
bet brief --days 8                        # the matchday recommendations
bet quality                               # reconciliation and coverage checks
bet lineup --team bayern_munich           # predicted XI, formation, rotation
bet fetch-news                            # pull news from several feeds and extract
bet watch                                 # T-60 line-up check and repricing
bet dashboard --out board.html            # static HTML overview
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
| FBref | per-match player lines (shots, xG, tackles, cards, minutes) | match pages, not season totals — see below |
| kicker.de | predicted and confirmed XIs | forward-looking only; historical XIs come free from FBref |

football-data.co.uk also carries **shots, shots on target, corners, fouls and
cards** per match, in the same CSVs already downloaded for results and odds.
Corners and cards are priced far more loosely than 1X2, so these arrive at no
extra request and land in `team_match_stat`.

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
  players.py         canonical player ids; name matching across sources
  availability.py    absence impact, estimated from squad data
  quality.py         reconciliation, coverage, staleness
  lineups.py         start propensity, formation, predicted XI
  matchday.py        T-60 confirmed-line-up check and repricing
  recommend.py       the matchday brief
  dashboard.py       the static HTML dashboard
  viz.py             inline SVG primitives: pitch, bars, lines, scatter
  extract/           LLM extraction: Ollama (default) and Claude backends
  models/            Model contract, baselines, Dixon-Coles, promoted prior, props
  spatial/           shot maps, heatmaps, field tilt (dashboard, not features)
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

## Player props

The strategic reason this layer exists: 1X2 is the most efficient market in
football and a public-data model is unlikely to beat it. Player props are priced
far more loosely — there are hundreds per matchday and they need exactly the
player-level data most bettors never assemble. That is where a player pipeline
pays for itself, and it is not somewhere a team-level model can help.

A prop count is built from four estimated pieces:

```
expected = per-90 rate × (minutes / 90) × opponent factor × volume factor
```

**Per-90 rates are shrunk** toward the position-group mean by empirical Bayes,
with the shrinkage strength estimated from the league's between-player variance
rather than chosen. A striker with one good match is priced at 2.5 shots, not 5.

**Expected minutes is the largest driver and the one most often got wrong.** A
confirmed XI collapses that uncertainty entirely, which is why the line-up
matters more than any amount of form data.

**Counts are negative binomial, not Poisson.** Overdispersion moves mass out of
the middle into both tails, so the effect depends on where the line sits: near
the mean it lowers the overs, in the upper tail it raises them sharply. At an
expected 2.0 shots, Poisson prices over 5.5 at 1.7% and a negative binomial at
5.0% — a factor of three, on exactly the alternative lines a book is least
careful about.

**Volume factor comes from the Dixon-Coles engine**, so a prop and the 1X2 price
can never imply different things about the same fixture. Elasticity is below one:
a team expected to score twice as much does not take twice as many shots.

Read `sample_90s` in the output before trusting a price. A thin sample means the
rate is mostly prior, not evidence.

## Why per-match player rows, not season totals

FBref serves cumulative season tables updated live, so a scrape taken today
contains matches that had not been played on the date you want to backtest, and
there is no way to subtract them back out. Per-match rows carry the final
whistle as their `known_at`, so any past Saturday's per-90 rates rebuild exactly.

One match page carries both squads across several stat tables, so a season costs
~306 requests rather than ~500 player pages — and it yields the actual XI for
free, which is where historical line-ups come from.

## Spatial analysis — for looking at, not for the model

`bet.spatial` produces shot maps with distance and angle, zone heatmaps, field
tilt and shot-quality profiles. It feeds the dashboard and scouting questions,
deliberately not the forecast.

A binned heatmap is ~50 numbers per team per match against 306 matches a season.
Handing that many weakly-informative spatial features to a model on that little
data is a fast route to overfitting, and almost everything a touch map says about
attacking quality is already inside xG with less noise.

The exception is **field tilt** — a single well-sampled scalar measuring
territorial dominance in a way possession share does not, which is why it is the
one spatial quantity worth passing to a model.

What the shot profile is genuinely good for: two sides with identical season xG
are not equally good if one gets there by volume. `xg_per_shot`,
`share_in_box` and `mean_distance_m` tell them apart, and no aggregate xG total
will.

## The dashboard

`bet dashboard --out board.html` writes one self-contained HTML file — no server,
no build step, no network. Four tabs: overview, fixtures, model health, props.

**Click any fixture** and it opens into the detail: both formations drawn on a
pitch, each side's eleven with start propensity and per-90 stats, the bench,
who's unavailable, and the ranked scorelines.

**Expected scores are shown only when the model can actually call one.**
Football scorelines are flat — in an even match 0-0, 1-1 and 1-0 often sit within
a point of each other — so a score appears only if it's genuinely the mode
(≥9%) *and* clearly ahead of the runner-up (≥20% margin). Otherwise the slot is
blank, with the reason on hover. On a typical matchday most fixtures are blank.
That is the honest output, not a gap.

**It tells you where its numbers came from.** A banner at the top names the
sources and the age of each feed. If every source is a test fixture it says so
in red — *synthetic data, not a real forecast* — because a dashboard built from
fixtures looks exactly like one built from Bundesliga results, and that
resemblance is how a demo gets read as a prediction. If the newest row is more
than 8 days old it says that too.

**Player tables carry a `last` column** — days since that player actually
started, amber past 28 days. A per-90 rate carries no date, so without it a
striker who stopped playing in September sits in the table looking identical to
one who played on Saturday.

Encoding notes, since they're decisions rather than defaults:

- **1X2 is diverging, not categorical.** Home and away are opposite outcomes, so
  they take the diverging poles (blue ↔ red) with the draw on the neutral grey
  midpoint. Three arbitrary hues would imply the outcomes are unordered
  identities; they aren't.
- **A dashed ring** on a pitch marks a player the model is under 55% sure will
  start. Confirmed XIs are solid.
- **One hue per series.** The scoreline chart doesn't shade bars by value — that
  would burn the only free channel restating the length the bar already shows.
- Palette is the validated reference set, checked with a validator rather than
  by eye (categorical trio passes all-pairs; the home/away poles clear CVD
  separation at ΔE 21.6). Light mode throughout.

`--backtest-from 2019-08-01` fills the model-health tab with a calibration curve
and the RPS ranking against baselines. It's slow, so it's off by default — and
until you run it, that tab tells you plainly that nothing else in the dashboard
has been shown to be any good.

It deliberately shows what the system *doesn't* know alongside what it does —
whether an XI is confirmed or guessed and with what confidence, how much evidence
sits behind a prop (`90s`), whether market prices existed to compare against. A
dashboard of conclusions alone invites more confidence than the numbers deserve.

## Predicted line-ups

The fix for a specific, expensive bug: team strength was computed from
squad-wide per-90 rates weighted by **cumulative** minutes. A striker who played
every week until September and hasn't appeared since still holds a large share
of the season's minutes, so the model kept pricing the team as though he plays —
while the squad player who has started the last six matches barely registered.

Strength now comes from the eleven expected to start:

- **Start propensity** — an exponentially recency-weighted start rate
  (21-day half-life). A player who started the last five matches scores near
  one; one who hasn't featured in six weeks scores near zero, whatever he
  banked in August. In the test case, two players with ten starts each score
  0.09 and 0.55 depending purely on *when* those starts happened.
- **Formation** — inferred from who is actually on the pitch, since sources
  disagree on formation strings and FBref supplies none. The predicted shape
  constrains the XI: a side playing 3-4-3 fields three centre-backs, so the
  eleven can't just be the eleven highest propensities.
- **Rotation** — mean churn between consecutive line-ups. A manager who names
  the same team every week scores near zero; that number *is* the confidence in
  any predicted XI, and it's reported rather than hidden.

A confirmed XI replaces the prediction outright. It isn't evidence to blend with
a guess — it's the answer.

One bug worth recording: scaling an unavailable player's propensity to zero
wasn't enough to remove him. Sorting alone still returned him, so when every
forward was ruled out the top three were three ruled-out forwards and the
absence silently had no effect. Ineligible players now leave the pool entirely.

## The T-60 workflow

`bet watch` polls for fixtures about an hour from kickoff, fetches the confirmed
line-up, re-prices, and reports what moved. The "before" price is a fresh
prediction from the same model with confirmed line-ups switched off, so team
news is the *only* thing that differs between the two numbers.

The diff is the useful part. A price that barely moves means the model had
already priced the XI correctly and there's nothing to act on. A large move
means it was pricing a striker who's on the bench, and the new number is the one
to trust.

This does not assume you'll beat the market to the news — books move within
seconds of an XI dropping and suspend markets while they do.

## Automatic news collection

`bet fetch-news` pulls from several independent feeds (kicker, bundesliga.com,
Sportschau, Guardian — German and English, different kinds of publisher) and
hands each article to the extraction pipeline. Diversity is the point: a club
channel buries what it would rather not report, an aggregator lags, a tabloid
over-reports a knock.

A relevance gate keeps the local model off match reports and transfer gossip.
Articles naming two clubs with no clear subject are **not** attributed — a
preview mentioning both sides isn't team news about either, and guessing would
attach one club's injury list to its opponent.

## Player availability

Knowing a player is injured is easy. Knowing what it is *worth* is where most
injury-adjusted models quietly fall apart: someone picks a number ("drop attack
15% if the striker is out"), it is never checked against anything, and it
decalibrates every probability the model produces.

Nothing here is hand-tuned. An absence is costed from data already in the store:
what the player contributes per 90, what his *actual replacement* contributes
(the best available squad member in that position, not a league-average
abstraction), and the difference as a share of the team's output. That converts
directly into a shift in the Dixon-Coles attack parameter, since rates are
exponential — a 6% loss of attacking output becomes `attack + log(0.94)`.

Three properties the tests enforce, each of which caught a real bug during
development:

- **Impact tracks contribution.** Losing the first-choice striker costs ~25% of
  attack; losing a squad midfielder costs ~8%.
- **A fringe player's absence barely moves anything.** The XI is unchanged, so
  the forecast should be too. Without weighting by playing time, ruling out a
  reserve shifted the model as if a starter had gone.
- **No absence can improve a team.** With one keeper in the squad there is no
  same-position cover, so the fallback replacement is an outfielder with a far
  higher attacking contribution — which made ruling out the first-choice keeper
  *raise* expected goals until the delta was floored at zero.

Honest caveat: the market prices known injuries faster than any scraper. The
realistic value is not beating the market to team news; it is stopping your own
model pricing a fixture around a striker who is on the bench.

## Team news extraction (local LLM)

`bet news` reads an article and writes validated rows to `player_availability`.
Three passes:

1. **Extract** — schema-constrained generation turns prose into entries, each
   carrying the sentence it relied on.
2. **Judge** — a second call, given the real squad list, decides whether each
   entry is genuinely supported. The extractor's incentives are wrong: asked to
   find team news, a model will find team news even in an article containing
   none.
3. **Resolve** — accepted names are matched against the actual squad. An
   unresolvable name is dropped, never guessed.

Everything rejected is kept, so a quiet week is distinguishable from a broken
extractor.

**Ollama is the default** (`qwen2.5:7b-instruct`). Extraction is high-volume and
low-difficulty, which is exactly where a small local model wins: zero marginal
cost, nothing leaves the machine, no rate limit. What makes it reliable is
Ollama's `format` parameter, which constrains generation to a JSON schema at
decode time — without it a 7B model returns markdown fences often enough to be
unusable. Qwen is the default because German-language team news is a meaningful
share of the sources and it handles non-English input better than similarly
sized Llama variants.

```bash
ollama serve && ollama pull qwen2.5:7b-instruct
```

`ClaudeBackend` implements the same interface for the occasional document worth
paying for. The judge pass matters *more* with a small model, not less.

## The matchday brief

`bet brief` is the layer everything else feeds. Match probabilities with
absence adjustments, fair odds, value bets where market prices exist, prop
prices, and the absences behind each adjustment — with every EV net of margin
and betting tax.

Two design commitments. A brief states its own uncertainty: thin prop samples
are suppressed, and a model probability more than 10 points above the market is
flagged as *probable model error, not value*. And `--narrate` is cosmetic only —
the local model is handed finished numbers and asked to read them out. It never
computes anything or decides what is recommended.

## Data quality

`bet quality` after every ingest. Scrapers do not fail loudly: a site changes
its markup and the parser returns empty tables, a column is renamed and a stat
silently becomes null — and the model keeps running, quietly getting worse.

The strongest check is redundancy. `match_result` is keyed on
**(match_id, source)**, so football-data.co.uk and OpenLigaDB results coexist
and can be compared — two scrapes either agree or one is wrong. Keyed on
`match_id` alone, the second ingest silently overwrote the first and the
disagreement was invisible; that was a real bug, found by a test. Reads pick a
source by fixed preference so backtests stay reproducible regardless of ingest
order.

Also checked: impossible odds, book sums below 1.0 (an "arbitrage" is far more
likely a parser bug than a real price), per-season coverage gaps, staleness, and
teams with too few appearances to be real — the signature of a name-resolution
failure splitting one club in two.

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

- Live odds ingestion and +EV alerting
- A scheduler — `bet watch` is one polling pass, meant for cron
- Line-up *fetching* — the T-60 workflow and the kicker parser exist, but the
  parser is unverified against live HTML (see below)
- Prop backtesting against historical prop lines (no free source carries them)
- Bayesian hierarchical variant, for parameter uncertainty that Kelly can use

## A warning about the example

`dashboard-example.html` in this repo is generated from the **synthetic test
fixtures**, not Bundesliga data. The player names come from a hardcoded surname
pool; the stats are Poisson draws. The page says so in red at the top.

Nothing in this project has yet been run against real data, because every source
(football-data.co.uk, FBref, Understat, ClubElo, OpenLigaDB) is unreachable from
the environment it was built in. The adapters are written and unit-tested
against fixture-shaped payloads; the first real `bet ingest` is still the moment
of truth.

## Testing

```bash
make test
```

326 tests, no network required. Synthetic seasons are generated from known team
strengths, so models are checked for recovering the truth rather than merely for
running without raising.
