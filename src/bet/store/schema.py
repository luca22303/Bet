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
CREATE TABLE IF NOT EXISTS match_result (
    match_id     VARCHAR PRIMARY KEY,
    home_goals   INTEGER NOT NULL,
    away_goals   INTEGER NOT NULL,
    outcome      VARCHAR NOT NULL,   -- H / D / A
    ht_home      INTEGER,
    ht_away      INTEGER,
    known_at     TIMESTAMP NOT NULL
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
    carries             DOUBLE,
    tackles             DOUBLE,
    interceptions       DOUBLE,
    blocks              DOUBLE,
    fouls               DOUBLE,
    yellow_cards        DOUBLE,
    red_cards           DOUBLE,
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

CREATE INDEX IF NOT EXISTS idx_pms_match ON player_match_stat (match_id);
CREATE INDEX IF NOT EXISTS idx_pms_player ON player_match_stat (player_id, known_at);
CREATE INDEX IF NOT EXISTS idx_lineup_match ON lineup (match_id);
CREATE INDEX IF NOT EXISTS idx_avail_player ON player_availability (player_id, known_at);

CREATE INDEX IF NOT EXISTS idx_match_kickoff ON match (kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_quote (match_id, market);
CREATE INDEX IF NOT EXISTS idx_shot_match ON shot (match_id);
CREATE INDEX IF NOT EXISTS idx_rating_team ON team_rating (team_id, valid_from);
"""

# Tables that carry point-in-time facts, checked by the leakage guard.
PIT_TABLES = ("match", "match_result", "odds_quote", "team_rating", "shot",
              "player", "player_match_stat", "lineup", "player_availability")
