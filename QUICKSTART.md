# Starting RueBet with live data

Every command below was run end to end. The only step that could not complete
is the network fetch itself, because the environment this was built in blocks
the data sources — which is exactly the step that will work on your machine.

## 1. Start it

On macOS or Linux:

```bash
git clone https://github.com/luca22303/Bet.git
cd Bet
git checkout claude/awesome-davinci-fesnx3
sh start.sh --refresh
```

On Windows, run the same three git commands, then `start.bat --refresh` — or
just double-click `start.bat` in Explorer.

That is the whole install. The script finds a suitable Python, creates a virtual
environment, installs everything into it, and opens
`http://127.0.0.1:8765` in your browser. First run takes about a minute; after
that it starts immediately.

**You do not need `pip` on your PATH.** macOS often ships only `pip3`, and a
Windows install frequently leaves `pip` unexported even when `python` works, so
every install step here goes through `python -m pip` instead. If Python itself
is missing, the script says which command installs it for your platform.

There is a **Refresh data** button on the page, so after the first run you never
need the terminal again.

Leave it running and bookmark the URL. Everything below is the same thing done
by hand, which is worth doing once for the backtest in step 4.

To use the `bet` command directly, activate the environment the script made:

```bash
source .venv/bin/activate
bet --help
```

On Windows the activate script is `.venv\Scripts\activate` instead.

The server binds to loopback only. The refresh endpoint fetches from the
internet and writes to the database with no authentication, which is fine for a
tool on your own machine and is not fine on a shared network — don't change
`--host` without putting an authenticating proxy in front of it.

## 2. First backfill — one source only

Start with football-data.co.uk alone. It is plain CSV, it is the source least
likely to have changed under the parser, and it carries both results **and**
historical closing odds, which is everything the backtest needs.

```bash
bet init
bet ingest --source football_data --seasons 2015-2026
```

Expect a few minutes and roughly 11 files. Then check what landed:

```bash
bet status
bet quality
```

`bet status` gives row counts and the point-in-time integrity check; `bet
quality` covers per-season coverage, odds sanity and staleness.


**What good looks like:** `bet status` ends with *"OK: no facts are visible
before they occurred"*, and `bet quality` shows ~306 matches per season with a
median closing overround around 4–7%.

**If club names raise errors,** the message names the club. Send it to me and
it is a one-line fix to `src/bet/teams.py`.

## 3. The number that decides everything

```bash
bet backtest --from 2018-08-01
```

This walks the model forward through seven seasons and scores it against
devigged closing prices. Read the `rps` column: lower is better, and **market**
is the benchmark.

- Model RPS **below** market → there is something here worth pursuing.
- Model RPS **above** market → the model has no betting edge, whatever any
  expected-value number elsewhere says.

Nothing else in this project matters until you have looked at that table. It is
also the first real test of whether the maths survives contact with actual
Bundesliga results rather than the synthetic data it was developed against.

## 4. The dashboard

`bet serve` from step 2 already gives you this at `http://127.0.0.1:8765`. For a
file to mail someone or serve statically:

```bash
bet live --out board.html
```

The banner at the top tells you what it is built from: **red** means the store
still holds only test fixtures, **amber** means the data is more than 8 days
old, and a plain line means it is current.

Click any fixture for the formations, both squads with their per-90 stats, and
the likely scorelines.

## 5. Add the richer sources

Once the above works, widen it. In this order, because it is also the order of
how likely each is to need a fix:

```bash
bet ingest --source clubelo
bet ingest --source openligadb --seasons 2015-2026
bet ingest --source understat --seasons 2015-2026
bet ingest --source understat --seasons 2023-2026 --shots
bet ingest --source fbref --seasons 2023-2026
```

In order: power ratings (which the promoted-team prior reads), fixtures plus a
second source for results, shot-level xG, per-match shots (slow — about 306
requests a season), and player stats with line-ups (slowest).


With two independent sources for results, `bet quality` can cross-check them —
two scrapes either agree on a scoreline or one of them is wrong.

FBref is the one most likely to need attention. If it returns nothing:

```bash
bet diagnose --source fbref --url https://fbref.com/en/matches/<some-match-page>
```

That prints what the page actually contains and saves the HTML. Send me the
output and the fix is precise rather than a guess.

## 6. Keep it current

If you leave `bet serve` running, press **Refresh data** on the page. For a
file on a schedule, `bet live` does fetch-and-render in one step.

Unattended, use cron rather than a loop — a scheduler survives a reboot:

```cron
0 7,19 * * *   cd /path/to/Bet && .venv/bin/bet live --out /var/www/board.html
0 12-18 * * 6  cd /path/to/Bet && .venv/bin/bet live --out /var/www/board.html
```

`bet live` exits non-zero when nothing landed, when the store is still
synthetic, or when the newest row is stale, so a broken refresh reaches you
through cron mail instead of silently serving yesterday's page.

## Optional: team news

Needs a local model. Extraction is high-volume and low-difficulty, which is
where a small local model beats a paid API.

```bash
ollama serve
ollama pull qwen2.5:7b-instruct

bet fetch-news
bet brief
```

## Optional: line-ups an hour before kickoff

```bash
bet watch
```

The useful output is the diff. A price that barely moves means the model had
already priced the eleven correctly; a large move means it was pricing someone
who is on the bench.

## Command reference

| Command | What it does |
|---|---|
| `bet serve` | run it locally and open it in a browser |
| `bet init` | create the database |
| `bet ingest --source X` | fetch one source |
| `bet live --out board.html` | fetch current season, rebuild the dashboard |
| `bet status` | row counts and the point-in-time check |
| `bet quality` | coverage, reconciliation, staleness |
| `bet backtest --from DATE` | walk forward and score against the market |
| `bet tune --from DATE` | pick the time-decay rate empirically |
| `bet dashboard --out X.html` | rebuild the page without fetching |
| `bet predict` / `bet brief` | terminal output for the coming fixtures |
| `bet lineup --team X` | predicted XI, formation, rotation |
| `bet props --stat shots` | player prop prices |
| `bet scout --team X` | shot profile and spatial summary |
| `bet watch` | T-60 line-up check and repricing |
| `bet diagnose --source X` | what a scraped page actually contains |

## Two things to expect

**Some parser will need a fix.** No adapter in this project has ever parsed a
real page — every source is unreachable from where it was built. The maths and
the plumbing are tested against data whose truth is known by construction; the
scrapers are tested against payloads written by the same hand that wrote the
parsers. `bet diagnose` exists for exactly this.

**The backtest may say there is no edge.** That is a real result, not a failure
of the setup. Beating a devigged closing line with public data is hard, and the
honest version of this project tells you so rather than showing you a confident
number.
