"""DuckDB schema for the point-in-time fact store.

The one rule this schema exists to enforce: every fact carries `known_at`, the
instant it became knowable. Backtests read the world through
`WHERE known_at <= as_of`, so a model can never be trained on a number that did
not exist yet.

This is not a nicety. Leaking post-match information into a backtest is the
single most common way a betting model appears profitable and is not, and the
leak is invisible in the output. It has to be structural.
"""

from __future__ import annotations

DDL = """
-- Raw payloads, archived before parsing. Keeping these means a source changing
-- its markup costs a re-parse rather than a re-scrape of a decade of history.
CREATE TABLE IF NOT EXISTS raw_document (
    doc_id       VARCHAR PRIMARY KEY,
    source       VARCHAR NOT NULL,
    url          VARCHAR NOT NULL,
    fetched_at   TIMESTAMP NOT NULL,
    content_hash VARCHAR NOT NULL,
    path         VARCHAR NOT NULL
);

-- Fixtures. A fixture's existence is known well before it is played, so
-- known_at is the announcement, not the kickoff.
CREATE TABLE IF NOT EXISTS match (
    match_id     VARCHAR PRIMARY KEY,
    source       VARCHAR NOT NULL,
    league       VARCHAR NOT NULL,
    season       VARCHAR NOT NULL,
    kickoff_utc  TIMESTAMP NOT NULL,
    home_team_id VARCHAR NOT NULL,
    away_team_id VARCHAR NOT NULL,
    known_at     TIMESTAMP NOT NULL
);

-- Results, knowable only after the final whistle.
--
-- Keyed on (match_id, source), not match_id alone, so two independent scrapes
-- of the same match coexist instead of one overwriting the other. That
-- redundancy is the only strong data-quality check available here: two sources
-- either agree on a scoreline or one of them is wrong, and a schema that
-- silently resolves the conflict by last-write-wins can never tell you which.
-- Reads pick a source by preference; `bet quality` compares them.
CREATE TABLE IF NOT EXISTS match_result (
    match_id     VARCHAR NOT NULL,
    source       VARCHAR NOT NULL DEFAULT 'unknown',
    home_goals   INTEGER NOT NULL,
    away_goals   INTEGER NOT NULL,
    outcome      VARCHAR NOT NULL,   -- H / D / A
    ht_home      INTEGER,
    ht_away      INTEGER,
    known_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, source)
);

-- One row per book per selection per observation. `is_closing` marks the last
-- price before kickoff, which is the benchmark every model is measured against.
CREATE TABLE IF NOT EXISTS odds_quote (
    match_id      VARCHAR NOT NULL,
    book          VARCHAR NOT NULL,
    market        VARCHAR NOT NULL,  -- '1x2', 'ou25', ...
    selection     VARCHAR NOT NULL,  -- 'H' / 'D' / 'A' / 'over' / 'under'
    decimal_odds  DOUBLE NOT NULL,
    quoted_at     TIMESTAMP NOT NULL,
    is_closing    BOOLEAN NOT NULL DEFAULT FALSE,
    known_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, book, market, selection, quoted_at)
);

-- Time series of external power ratings (ClubElo and anything similar).
CREATE TABLE IF NOT EXISTS team_rating (
    team_id     VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,
    rating      DOUBLE NOT NULL,
    valid_from  DATE NOT NULL,
    valid_to    DATE,
    known_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (team_id, source, valid_from)
);

-- Shot-level expected goals. The cleanest available signal of team quality.
CREATE TABLE IF NOT EXISTS shot (
    shot_id     VARCHAR PRIMARY KEY,
    match_id    VARCHAR NOT NULL,
    team_id     VARCHAR NOT NULL,
    player_name VARCHAR,
    minute      INTEGER,
    x           DOUBLE,
    y           DOUBLE,
    xg          DOUBLE,
    body_part   VARCHAR,
    situation   VARCHAR,
    result      VARCHAR,
    known_at    TIMESTAMP NOT NULL
);

-- Model output, stored so calibration can be audited after the fact.
CREATE TABLE IF NOT EXISTS prediction (
    match_id   VARCHAR NOT NULL,
    model      VARCHAR NOT NULL,
    as_of      TIMESTAMP NOT NULL,
    p_home     DOUBLE NOT NULL,
    p_draw     DOUBLE NOT NULL,
    p_away     DOUBLE NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, model, as_of)
);

-- Team-level match stats: shots, corners, cards, fouls.
--
-- These ship inside the football-data.co.uk CSVs already being downloaded for
-- results and odds, at no extra request. Corners and cards in particular are
-- priced far more loosely than 1X2, so discarding them was throwing away the
-- softest markets available for free.
CREATE TABLE IF NOT EXISTS team_match_stat (
    match_id        VARCHAR NOT NULL,
    team_id         VARCHAR NOT NULL,
    source          VARCHAR NOT NULL,
    at_home         BOOLEAN NOT NULL,
    shots           INTEGER,
    shots_on_target INTEGER,
    corners         INTEGER,
    fouls           INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    known_at        TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, team_id, source)
);

-- Player registry. Squads turn over constantly, so a name appearing for the
-- first time is a signing, not an error.
CREATE TABLE IF NOT EXISTS player (
    player_id  VARCHAR PRIMARY KEY,
    full_name  VARCHAR NOT NULL,
    source     VARCHAR,
    source_id  VARCHAR,
    known_at   TIMESTAMP NOT NULL
);

-- Per-match player lines, NOT season totals.
--
-- This distinction is the whole reason the table looks like this. FBref serves
-- cumulative season totals that are updated live, so a scrape today carries
-- next week's matches inside it and is unusable for any backtest before the
-- scrape date. Per-match rows carry known_at = final whistle, so a per-90 rate
-- can be rebuilt as it stood on any past Saturday.
CREATE TABLE IF NOT EXISTS player_match_stat (
    match_id            VARCHAR NOT NULL,
    player_id           VARCHAR NOT NULL,
    team_id             VARCHAR NOT NULL,
    source              VARCHAR NOT NULL,
    position            VARCHAR,
    started             BOOLEAN,
    minutes             DOUBLE,
    goals               DOUBLE,
    assists             DOUBLE,
    shots               DOUBLE,
    shots_on_target     DOUBLE,
    xg                  DOUBLE,
    npxg                DOUBLE,
    xa                  DOUBLE,
    passes_completed    DOUBLE,
    passes_attempted    DOUBLE,
    progressive_passes  DOUBLE,
    touches             DOUBLE,
    -- FBref's own pitch-zone breakdown of that total: coarse (five zones),
    -- but real touch-location data, unlike anything derived from tracking.
    -- Feeds the zone heatmap; None on any row ingested before this existed.
    touches_def_pen     DOUBLE,
    touches_def_third   DOUBLE,
    touches_mid_third   DOUBLE,
    touches_att_third   DOUBLE,
    touches_att_pen     DOUBLE,
    carries             DOUBLE,
    tackles             DOUBLE,
    interceptions       DOUBLE,
    blocks              DOUBLE,
    fouls               DOUBLE,
    yellow_cards        DOUBLE,
    red_cards           DOUBLE,
    -- FBref's Shooting, Possession, Defensive Actions, Misc and Goalkeeper
    -- tables. None on a row ingested before these existed, same as the
    -- touch zones above -- "never ingested", not "a real zero".
    avg_shot_distance    DOUBLE,   -- metres, converted from FBref's yards
    dribbles_attempted   DOUBLE,
    dribbles_completed   DOUBLE,
    dribbles_tackled     DOUBLE,   -- dribble attempted, opponent won the ball
    tackles_won          DOUBLE,
    challenges_attempted DOUBLE,   -- tackle attempts specifically vs a dribble
    challenges_lost      DOUBLE,   -- attempted, opponent beat the challenge
    aerials_won          DOUBLE,
    aerials_lost         DOUBLE,
    fouls_drawn          DOUBLE,
    recoveries           DOUBLE,
    gk_shots_faced       DOUBLE,
    gk_goals_against     DOUBLE,
    gk_saves             DOUBLE,
    gk_save_pct          DOUBLE,
    known_at            TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, player_id, source)
);

-- Confirmed and predicted line-ups.
--
-- The confirmed XI, roughly an hour before kickoff, is the only genuinely
-- time-sensitive signal in this system. Predicted line-ups published earlier
-- are stored alongside it with their own known_at and a lower confidence, so a
-- backtest can ask what was knowable at any lead time.
CREATE TABLE IF NOT EXISTS lineup (
    match_id      VARCHAR NOT NULL,
    player_id     VARCHAR NOT NULL,
    team_id       VARCHAR NOT NULL,
    source        VARCHAR NOT NULL,
    is_starter    BOOLEAN NOT NULL,
    is_confirmed  BOOLEAN NOT NULL DEFAULT FALSE,
    shirt_number  INTEGER,
    formation     VARCHAR,
    known_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, player_id, source, known_at)
);

-- Injury and suspension reports. Append-only: a player's status changes over
-- time and the history is what lets a backtest see what was known then.
CREATE TABLE IF NOT EXISTS player_availability (
    player_id        VARCHAR NOT NULL,
    team_id          VARCHAR NOT NULL,
    source           VARCHAR NOT NULL,
    status           VARCHAR NOT NULL,   -- OUT / DOUBTFUL / FIT
    reason           VARCHAR,
    expected_return  DATE,
    confidence       DOUBLE,
    known_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id, source, known_at)
);

-- There are deliberately no secondary indexes here. See DROPPED_INDEXES.
"""

# Secondary indexes this schema used to create, dropped from any database that
# still has them.
#
# They were where a refresh died: "Failed to delete all rows from index. Only
# deleted 0 out of 36 rows", which invalidates the entire database rather than
# failing the one statement. The failure names the `match` table's two indexed
# columns, and `match (kickoff_utc)` is a *non-unique* index over a column where
# a whole matchday shares one timestamp -- nine fixtures, one key. Updating a
# row deletes its old version from every index, so an upsert touches them even
# though it no longer issues a DELETE of its own.
#
# They also earned nothing. Measured over eleven seasons (3,366 matches, 74,052
# player-match rows), every read this project makes is within noise of the same
# query without them -- DuckDB is columnar and answers these by scanning with
# zone maps -- while ingestion ran about 40% slower with them present. An index
# that costs write throughput, returns no read speed, and can invalidate the
# database is not a trade worth keeping.
#
# Primary keys stay: they are unique, so they do not hit the duplicate-key path,
# and `INSERT ... ON CONFLICT` needs them to detect a collision at all.
#
# Dropping them also repairs a database already carrying a broken index, which
# is why this runs on every `init_schema` rather than once.
DROPPED_INDEXES = (
    "idx_tms_match", "idx_pms_match", "idx_pms_player", "idx_lineup_match",
    "idx_avail_player", "idx_match_kickoff", "idx_odds_match", "idx_shot_match",
    "idx_rating_team",
)


# Columns added to an existing table after it first shipped. `CREATE TABLE IF
# NOT EXISTS` does nothing to a table that already exists, so a store created
# before one of these was added would otherwise never gain it -- the zone
# heatmap columns, say, silently missing from a database ingested months ago
# with no error to say why the heatmap tab stays empty.
ADDED_COLUMNS = (
    ("player_match_stat", "touches_def_pen", "DOUBLE"),
    ("player_match_stat", "touches_def_third", "DOUBLE"),
    ("player_match_stat", "touches_mid_third", "DOUBLE"),
    ("player_match_stat", "touches_att_third", "DOUBLE"),
    ("player_match_stat", "touches_att_pen", "DOUBLE"),
    ("player_match_stat", "avg_shot_distance", "DOUBLE"),
    ("player_match_stat", "dribbles_attempted", "DOUBLE"),
    ("player_match_stat", "dribbles_completed", "DOUBLE"),
    ("player_match_stat", "dribbles_tackled", "DOUBLE"),
    ("player_match_stat", "tackles_won", "DOUBLE"),
    ("player_match_stat", "challenges_attempted", "DOUBLE"),
    ("player_match_stat", "challenges_lost", "DOUBLE"),
    ("player_match_stat", "aerials_won", "DOUBLE"),
    ("player_match_stat", "aerials_lost", "DOUBLE"),
    ("player_match_stat", "fouls_drawn", "DOUBLE"),
    ("player_match_stat", "recoveries", "DOUBLE"),
    ("player_match_stat", "gk_shots_faced", "DOUBLE"),
    ("player_match_stat", "gk_goals_against", "DOUBLE"),
    ("player_match_stat", "gk_saves", "DOUBLE"),
    ("player_match_stat", "gk_save_pct", "DOUBLE"),
)

MIGRATIONS = "\n".join(
    [f"DROP INDEX IF EXISTS {name};" for name in DROPPED_INDEXES]
    + [f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {kind};"
       for table, column, kind in ADDED_COLUMNS]
)

# Tables that carry point-in-time facts, checked by the leakage guard.
PIT_TABLES = ("match", "match_result", "odds_quote", "team_rating", "shot",
              "player", "player_match_stat", "lineup", "player_availability",
              "team_match_stat")
