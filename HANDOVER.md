# Handover — 2026-10-26

For whoever (or whichever Claude session) picks this up next, most likely a
**local** session on Luca's own machine where the real data sources and his
real DuckDB file are actually reachable — this cloud sandbox is not: it has
no network route to football-data.co.uk, OpenLigaDB, ClubElo, FBref,
Sofascore or kicker.de at all (confirmed via the sandbox's own proxy status
endpoint — `fbref.com:443` gets a 403 at the CONNECT/policy layer, before the
request ever reaches the real site). Everything below about Luca's real data
came from him pasting terminal output back, not from this session running it.

Repo: `luca22303/Bet`. Only branch: `claude/awesome-davinci-fesnx3` (also
HEAD/default — there is no `main`). User: Luca Rüffer
(luca.rueffer@scalors.com), running Windows/PowerShell locally at
`C:\Users\LucaRüfferScalorsGmb\Bet` (`~/Bet` in his shell).

## State as of this commit

Pushed this session, in order:
- `b7e9a25` — FBref 403 fix (realistic browser UA) + the first point-in-time
  leak fix (football_data.py was treating UK-local kickoff times as UTC).
- `f415318` — new read-only Kickbase squad advisor (`bet kickbase-advice`),
  fully tested on synthetic data (client.py's real API shape is an unverified
  community guess — see its module docstring).
- `7c50911` — dropped `clubelo` from the default sources for `bet serve`/
  `bet live`/`start.bat`/`start.sh` (ClubElo kept timing out for Luca; still
  usable manually via `--sources ...,clubelo`).
- `2f5ced8` — fixed a second real timezone bug, this time in
  `openligadb.py`'s kickoff-parsing fallback (used when `matchDateTimeUTC` is
  missing from an entry): it was reading `matchDateTime` (German local time)
  as if already UTC, the exact mistake the function's own docstring warns
  against. Also added a fuller set of browser-realistic headers (Accept,
  Accept-Language, Connection, Upgrade-Insecure-Requests) to the shared
  `Source.__init__`, on top of the realistic UA from `b7e9a25`.

Full test suite (`python3.13 -m pytest`, ~638 tests) was green before every
push above.

## Open problem #1 — point-in-time leakage persists, my fix for it did NOT work

Luca's real local store (deep history, 2015-16 through 2026-27, seeded from
his own `start.bat --refresh` runs over previous sessions) reports, both
before and after the `2f5ced8` fix:

```
point-in-time integrity:
       table  violations
match_result         301
  odds_quote       17322
FAIL: 17623 rows are knowable before they should be.
```

**I was wrong about the cause.** I assumed the `openligadb.py` fallback bug
(fixed in `2f5ced8`) was the trigger, reasoning that OpenLigaDB's corrected
kickoff would overwrite football_data's kickoff in the shared `match` table
and the numbers would drop. Luca re-ran `bet ingest --source openligadb
--seasons 2015-2027` (cache=True, so it re-parsed already-downloaded JSON
with the fixed code, no new network calls) and then `bet check` —
**the violation counts did not change at all, to the row.** That means
either `matchDateTimeUTC` was present on every real entry already (the
buggy fallback path never actually fired on his data), or something else
entirely is going on. The openligadb fix is still a real, confirmed bug
(demonstrated by a now-corrected unit test that had encoded the wrong
behavior as expected), but it is very likely **not** what's producing these
17,623 rows.

### My current best (unconfirmed) theory

`match` is upserted keyed on `match_id` **alone** — not `(match_id, source)`
like `match_result` and `odds_quote` are. See `football_data.py` and
`openligadb.py`, both call:
```python
self.store.upsert("match", pd.DataFrame(matches).drop_duplicates("match_id"), ["match_id"])
```
So whichever source ingests *last* silently overwrites `match.kickoff_utc`
for a shared `match_id` (same bug class the README already documents for an
older `match_result` incident, just never fixed for `match` itself).

Meanwhile, `known_at` on `match_result`/`odds_quote` rows is computed **once,
at ingest time, from that source's own kickoff estimate**, and baked into
the row permanently — it is never recomputed later even if a *different*
source subsequently overwrites the shared `match.kickoff_utc` to a different
value. The leakage check always compares against the *current* (i.e. latest
overwrite's) `match.kickoff_utc`:
```sql
SELECT COUNT(*) FROM match_result r JOIN match m USING (match_id)
WHERE r.known_at < m.kickoff_utc
-- and, for odds_quote:
WHERE o.known_at > m.kickoff_utc AND o.is_closing
```
So if football_data's own kickoff estimate and OpenLigaDB's kickoff estimate
disagree for the same `match_id` — even by minutes — whichever one *didn't*
write last has its own, previously-correct `known_at` compared against a
*different* kickoff basis than the one it was computed from, and can flip
into a false violation in either direction depending on which way the
disagreement goes. `football_data.py`'s defaulted-to-21:00-UK-local
placeholder (for pre-2019 seasons whose CSV has no `Time` column — see its
own docstring) is a prime suspect for disagreeing with OpenLigaDB's precise
real kickoff for the same old matches, in exactly the two opposite
directions that would explain why `match_result` and `odds_quote` violate
for different (one would guess disjoint) subsets of rows.

**This is a plausible explanation, not a confirmed one.** It was not
verified against Luca's real data before this handover was written — only
reasoned through by reading the code. The next session should confirm it
before trying another fix cycle, e.g. by querying Luca's actual
`data/db/bet.duckdb` (which this sandbox cannot do — no access to his
machine) for a handful of violating rows, comparing `match_result.known_at`
per source against `match.kickoff_utc`, to see the actual direction and size
of the disagreement. Something in this shape, run on his machine:

```python
import duckdb
con = duckdb.connect("data/db/bet.duckdb", read_only=True)
print(con.execute("""
    SELECT r.match_id, r.source, r.known_at, m.kickoff_utc,
           r.known_at - m.kickoff_utc AS gap
    FROM match_result r JOIN match m USING (match_id)
    WHERE r.known_at < m.kickoff_utc
    LIMIT 10
""").fetchdf())
print(con.execute("""
    SELECT o.match_id, o.book, o.known_at, m.kickoff_utc,
           o.known_at - m.kickoff_utc AS gap
    FROM odds_quote o JOIN match m USING (match_id)
    WHERE o.known_at > m.kickoff_utc AND o.is_closing
    LIMIT 10
""").fetchdf())
```

If the theory holds, the real fix is architectural, not a one-line timezone
patch: either give `match` a `source` column too and stop collapsing kickoff
across sources, or pick one source as the authoritative kickoff for a given
`match_id` and never let a lower-precision guess (the 21:00 placeholder)
override or get overridden inconsistently, or recompute `known_at` from the
*current* `match.kickoff_utc` at query time rather than baking it in at
ingest time. Don't commit to one of these without first confirming the
actual failure mode above — I got the first guess wrong once already this
session; verify before fixing again.

## Open problem #2 — FBref still returns 403

Even after the UA fix (`b7e9a25`) and the fuller header set (`2f5ced8`),
Luca's real run still gets:
```
! schedule 2026-27: 403 Client Error: Forbidden for url: https://fbref.com/en/comps/20/2026-2027/schedule/2026-2027-Bundesliga-Scores-and-Fixtures
```
This sandbox cannot reach fbref.com at all to iterate further (see above) —
every future attempt here would be another blind guess. Two live
possibilities, not yet distinguished:
- A Cloudflare JS/managed challenge, which no header combination fixes with
  plain `requests` — would need a real browser (Playwright is already
  available as a dependency pattern elsewhere in sandboxed dev, but nothing
  in this repo uses it yet) to pass.
- An IP-level temporary soft-ban from Luca's own earlier scraping attempts,
  which usually self-expires given enough time.

**A local Claude Code session on Luca's own machine could actually test this
live** (real network, real response headers/body) instead of guessing blind
from here — I suggested this to him and he may want to do that for both this
and the point-in-time puzzle above.

## Standing project discipline (for whoever continues)

- Full test suite (`python3.13 -m pytest`, use `python3.13` specifically in
  this sandbox — `/usr/local/bin/python3` has no pytest installed) must be
  green before any push.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_018K3iomn9BrwNvBNcJavUsQ
  ```
- No PR unless explicitly asked.
- Never fabricate data; be explicit when something is a synthetic demo vs.
  real data, and when a claim ("this fixes it") hasn't actually been
  confirmed against Luca's real data yet (see problem #1 above — say so
  plainly rather than implying confidence you don't have).
