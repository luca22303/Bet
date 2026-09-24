"""Parser robustness against markup that is not what we assumed.

Neither FBref nor kicker is reachable from the environment this project was
built in, so no parser here has met its real page. The response is not to guess
harder at selectors but to parse on content, and to prove that against several
plausible markups rather than the single one the author had in mind.

Every page below is synthetic. Passing these does not mean the parsers work
live; it means a markup change of the kind that normally breaks a scraper does
not break these.
"""

import json

import pytest

from bet.ingest.fbref import (
    drop_footer,
    extract_kickoff,
    extract_player_id,
    extract_teams,
    flatten_columns,
    iter_tables,
    looks_like_player_table,
    squad_key,
    _is_not_a_player,
)
from bet.ingest.kicker import (
    extract_embedded_json,
    parse_lineup_page,
    plausible_formation,
    _is_name_like,
)


# --------------------------------------------------------------- fbref pages


def _player_table(table_id, prefix, count=12):
    rows = "".join(
        f'<tr><th><a href="/en/players/{abs(hash(prefix)) % 0xffff:04x}{i:04x}/x">'
        f'{prefix} P{i}</a></th><td>{"GK" if i == 0 else "DF" if i < 5 else "MF"}</td>'
        f"<td>90</td><td>0</td><td>1</td><td>0.10</td><td>2</td></tr>"
        for i in range(count))
    attr = f' id="{table_id}"' if table_id else ""
    return (f"<table{attr}><thead>"
            '<tr><th colspan="2"></th><th></th><th colspan="2">Performance</th>'
            '<th>Expected</th><th>Tackles</th></tr>'
            "<tr><th>Player</th><th>Pos</th><th>Min</th><th>Gls</th><th>Sh</th>"
            "<th>xG</th><th>Tkl</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "<tfoot><tr><th>12 Players</th><td></td><td>1080</td><td>0</td>"
            "<td>12</td><td>1.2</td><td>24</td></tr></tfoot></table>")


CURRENT_STYLE = (
    "<html><head><title>Bayern Munich vs Wolfsburg Match Report | FBref</title></head>"
    '<body><span class="venuetime" data-venue-epoch="1724510400"></span>'
    + _player_table("stats_abc12345_summary", "BAY")
    + "<div><!--" + _player_table("stats_def67890_summary", "WOL") + "--></div>"
    + "</body></html>")

REDESIGNED_IDS = (
    '<html><head><meta property="og:title" content="Bayern Munich vs. Wolfsburg '
    'Match Report"></head><body><time datetime="2024-08-24T15:30:00Z"></time>'
    + _player_table("player_stats_home_2024", "BAY")
    + _player_table("player_stats_away_2024", "WOL") + "</body></html>")

NO_IDS = (
    "<html><head><title>Bayern Munich vs Wolfsburg Match Report</title></head>"
    '<body><time datetime="2024-08-24T15:30:00Z"></time>'
    + _player_table(None, "BAY") + _player_table(None, "WOL") + "</body></html>")


@pytest.mark.parametrize("page", [CURRENT_STYLE, REDESIGNED_IDS, NO_IDS],
                         ids=["current", "redesigned-ids", "no-ids"])
def test_fbref_parses_regardless_of_id_convention(store, page):
    """Content decides which tables matter, not the id format.

    The previous parser matched `id="stats_<hash>_summary"` and would have
    returned nothing the moment FBref changed that convention -- and since
    nobody here can open the live page, an id format is a guess.
    """
    from bet.ingest.base import IngestResult
    from bet.ingest.fbref import FBrefSource

    source = FBrefSource(store, raw_dir="/tmp", delay=0)
    players, stats = source._parse_match_page(
        page, "bundesliga", "2024-25", IngestResult(source="fbref"))

    assert len(stats) == 24                       # 12 per side, no totals row
    assert len({s["team_id"] for s in stats}) == 2
    assert sum(1 for s in stats if s["started"]) == 22
    assert all(s["minutes"] == 90.0 for s in stats)
    assert all(s["xg"] is not None for s in stats)


@pytest.mark.parametrize("html,expected", [
    ("<title>Bayern Munich vs Wolfsburg Match Report | FBref</title>",
     ("Bayern Munich", "Wolfsburg")),
    ('<meta property="og:title" content="Bayern Munich vs. Wolfsburg Match Report">',
     ("Bayern Munich", "Wolfsburg")),
    ("<h1><span>Bayern Munich vs Wolfsburg Match Report</span></h1>",
     ("Bayern Munich", "Wolfsburg")),
])
def test_team_names_have_several_independent_routes(html, expected):
    """Any one can break on a redesign, so all are tried."""
    assert extract_teams(html) == expected


def test_team_extraction_reports_failure_rather_than_guessing():
    assert extract_teams("<html><p>nothing</p></html>") == (None, None)


@pytest.mark.parametrize("html", [
    '<span data-venue-epoch="1724510400"></span>',
    '<time datetime="2024-08-24T15:30:00Z"></time>',
    '<span data-venue-date="2024-08-24"></span>',
])
def test_kickoff_has_several_independent_routes(html):
    assert extract_kickoff(html) is not None


def test_table_footer_is_not_ingested_as_a_player():
    """FBref closes each table with a totals row reading "12 Players".

    It parses as an ordinary row and matches no obvious keyword, so without
    handling it each squad gains a phantom player carrying the team's summed
    minutes -- which then distorts every per-90 rate built from them.
    """
    assert "<tfoot" not in drop_footer(_player_table("t", "BAY"))
    assert _is_not_a_player("12 Players")
    assert _is_not_a_player("Total")
    assert not _is_not_a_player("Harry Kane")


def test_missing_parser_backend_is_loud_not_silent():
    """pandas.read_html needs lxml or bs4.

    The earlier code caught ImportError alongside a malformed-table ValueError,
    which turned a missing dependency into an ingest that quietly returned
    nothing -- the exact silent-scraper failure this project spends a module
    trying to detect.
    """
    import inspect

    from bet.ingest import fbref
    source = inspect.getsource(fbref._read_table)
    assert "ImportError" in source
    assert "raise RuntimeError" in source
    assert "except (ValueError, ImportError)" not in inspect.getsource(fbref)


def test_a_tfoot_does_not_shift_every_row_ids_extraction(store):
    """`frame` is parsed with the tfoot already dropped, so its row count no
    longer includes FBref's totals row -- matching player ids against <tr>
    blocks taken from the raw, tfoot-including HTML shifted every row by one,
    with the tfoot's own row wrongly claimed as the last player's. Invisible
    before because every table on a page carries a tfoot, so two tables for
    the same players were shifted identically and still merged by accident;
    a table without one (like the goalkeeper table below) breaks that."""
    from bet.ingest.base import IngestResult
    from bet.ingest.fbref import FBrefSource

    page = ("<html><head><title>Bayern Munich vs Wolfsburg Match Report | FBref"
            "</title></head><body>"
            '<span class="venuetime" data-venue-epoch="1724510400"></span>'
            + _player_table("stats_abc12345_summary", "BAY", count=3)
            + _player_table("stats_def67890_summary", "WOL", count=3)
            + "</body></html>")

    source = FBrefSource(store, raw_dir="/tmp", delay=0)
    _, stats = source._parse_match_page(
        page, "bundesliga", "2024-25", IngestResult(source="fbref"))

    bay = [s for s in stats if s["team_id"] == "bayern_munich"]
    assert len(bay) == 3
    # A stable fbref: id on every player, none of them a name-slug fallback --
    # the tell that a row's link got claimed by the tfoot's totals row instead.
    assert all(s["player_id"].startswith("fbref:") for s in bay)
    assert len({s["player_id"] for s in bay}) == 3


def test_squad_key_groups_by_hash_then_by_position():
    assert squad_key("stats_abc12345_summary", 0) == "abc12345"
    assert squad_key("stats_abc12345_passing", 3) == "abc12345"
    assert squad_key(None, 2) == "position:2"


def test_player_id_prefers_the_stable_source_id():
    row = '<tr><td><a href="/en/players/d70ce98e/Harry-Kane">Harry Kane</a></td></tr>'
    assert extract_player_id(row, "Harry Kane") == "fbref:d70ce98e"
    assert extract_player_id("<tr><td>Harry Kane</td></tr>", "Harry Kane") == "harry_kane"


def test_grouped_columns_keep_their_prefix():
    """Without it, passing 'Att' and take-on 'Att' silently collide."""
    import pandas as pd
    frame = pd.DataFrame([[1, 2, 3]], columns=pd.MultiIndex.from_tuples(
        [("Performance", "Gls"), ("Passes", "Att"), ("Take-Ons", "Att")]))
    assert list(flatten_columns(frame).columns) == [
        "performance_gls", "passes_att", "take_ons_att"]


# -------------------------------------------------------------- kicker pages


ELEVEN = ["Neuer", "Kimmich", "Upamecano", "Kim", "Davies", "Goretzka",
          "Pavlovic", "Sane", "Musiala", "Olise", "Kane"]
OTHER = ["Kobel", "Ryerson", "Schlotterbeck", "Anton", "Svensson", "Gross",
         "Sabitzer", "Adeyemi", "Brandt", "Gittens", "Guirassy"]


def _kicker_json_page():
    payload = {"props": {"pageProps": {"match": {
        "home": {"name": "Bayern München",
                 "lineup": [{"name": n} for n in ELEVEN]},
        "away": {"name": "Borussia Dortmund",
                 "lineup": [{"name": n} for n in OTHER]}}}}}
    return ("<html><body>Aufstellung bestätigt 4-2-3-1"
            f'<script id="__NEXT_DATA__" type="application/json">'
            f"{json.dumps(payload)}</script></body></html>")


def _kicker_markup_page():
    def block(team, players, formation):
        names = "".join(f'<li class="player-name_x7f2">{n}</li>' for n in players)
        return (f'<div class="kick__lineup_a91"><span class="team-name_b3">{team}</span>'
                f"<span>{formation}</span><ul>{names}</ul></div>")
    return ("<html><body>Offizielle Aufstellung"
            + block("Bayern München", ELEVEN, "4-2-3-1")
            + block("Borussia Dortmund", OTHER, "4-3-3") + "</body></html>")


def _kicker_text_page():
    return ("<html><body><p>4-2-3-1</p><span>Bayern München</span>"
            + "".join(f"<span>{n}</span>" for n in ELEVEN)
            + "<p>4-3-3</p><span>Borussia Dortmund</span>"
            + "".join(f"<span>{n}</span>" for n in OTHER) + "</body></html>")


@pytest.mark.parametrize("page,strategy", [
    (_kicker_json_page(), "embedded-json"),
    (_kicker_markup_page(), "markup"),
    (_kicker_text_page(), "generic"),
], ids=["json", "markup", "text"])
def test_kicker_falls_back_through_its_strategies(page, strategy):
    """Three routes, most structure-independent last.

    kicker.de is unreachable from here, so the fallback chain is the whole
    defence: the strategy that works is reported, making a page that starts
    parsing differently visible rather than silent.
    """
    result = parse_lineup_page(page)
    assert result["strategy"] == strategy
    assert len(result["teams"]) == 2
    for team in result["teams"]:
        assert len(team["players"]) == 11
        assert "ü" in team["name"] or team["name"].isascii()


def test_kicker_reports_when_nothing_parses():
    result = parse_lineup_page("<html><body><p>no line-ups here</p></body></html>")
    assert result["teams"] == []
    assert result["strategy"] == "none"


def test_confirmed_is_distinguished_from_predicted():
    """A predicted XI is a guess with an error rate.

    Treating one as confirmed silently corrupts every downstream minutes
    estimate.
    """
    assert parse_lineup_page(_kicker_json_page())["confirmed"]
    assert not parse_lineup_page(_kicker_text_page())["confirmed"]


@pytest.mark.parametrize("text,expected", [
    ("Aufstellung 4-2-3-1 heute", "4-2-3-1"),
    ("spielt 3-4-3", "3-4-3"),
    ("Endstand 2-1", None),          # a score, not a formation
    ("24-08-2024", None),            # a date
    ("5-3-2 defensiv", "5-3-2"),
])
def test_formation_detection_requires_ten_outfield_players(text, expected):
    """The arithmetic is what separates a formation from a score or a date."""
    assert plausible_formation(text) == expected


def test_team_name_is_not_read_as_a_twelfth_player():
    """A bare `name` class pattern also matches `team-name`."""
    result = parse_lineup_page(_kicker_markup_page())
    for team in result["teams"]:
        assert team["name"] not in team["players"]
        assert len(team["players"]) == 11


@pytest.mark.parametrize("name,ok", [
    ("Harry Kane", True), ("Müller", True), ("Di María", True),
    ("Aufstellung", False), ("Trainer", False), ("12", False), ("Bank", False),
])
def test_section_headings_are_not_mistaken_for_players(name, ok):
    assert _is_name_like(name) is ok


def test_embedded_json_is_found_in_several_wrappers():
    for wrapper in ('<script id="__NEXT_DATA__">{"a":1}</script>',
                    'window.__NUXT__ = {"a":1};',
                    '<script type="application/ld+json">{"a":1}</script>'):
        assert extract_embedded_json(wrapper) == [{"a": 1}]


# ----------------------------------------------------------------- diagnose


def test_diagnose_passes_a_good_fbref_page():
    from bet.diagnose import diagnose_fbref
    findings = diagnose_fbref(CURRENT_STYLE)
    assert all(f.ok for f in findings), [str(f) for f in findings if not f.ok]


def test_diagnose_says_what_a_broken_page_contains():
    """The point of the command.

    When a parser fails live, the useful output is not "it failed" but what the
    page actually holds -- enough to correct the parser without seeing the site.
    """
    from bet.diagnose import diagnose_fbref
    findings = diagnose_fbref(
        "<html><head><title>Some Page</title></head><body>nothing</body></html>")
    failed = [f for f in findings if not f.ok]
    assert failed
    evidence = " ".join(e for f in failed for e in f.evidence)
    assert "Some Page" in evidence          # names the actual title
    assert "tried:" in evidence             # and what was attempted


def test_diagnose_reports_kicker_class_names():
    from bet.diagnose import diagnose_kicker
    findings = diagnose_kicker(
        '<html><body><div class="mod-lineup"><p>2-1</p></div></body></html>')
    evidence = " ".join(e for f in findings for e in f.evidence)
    assert "mod-lineup" in evidence         # the real class names
    assert "2-1" in evidence                # the digit runs it rejected


def test_diagnose_checks_every_kicker_strategy_not_just_the_winner():
    from bet.diagnose import diagnose_kicker
    steps = {f.step for f in diagnose_kicker(_kicker_json_page())}
    assert "strategy: embedded-json" in steps
    assert "strategy: markup" in steps
    assert "strategy: generic" in steps


def _possession_table(table_id, prefix, count=12):
    """A possession-table page fragment sharing the same player identities as
    `_player_table`, so its rows merge into the same records rather than
    creating twelve phantom extra players."""
    rows = "".join(
        f'<tr><th><a href="/en/players/{abs(hash(prefix)) % 0xffff:04x}{i:04x}/x">'
        f'{prefix} P{i}</a></th><td>90</td>'
        f"<td>{40 + i}</td><td>{5 + i % 3}</td><td>{10 + i}</td><td>{12 + i}</td>"
        f"<td>{8 + i}</td><td>{5 + i}</td></tr>"
        for i in range(count))
    attr = f' id="{table_id}"' if table_id else ""
    return (f"<table{attr}><thead>"
            '<tr><th colspan="2"></th><th colspan="6">Touches</th></tr>'
            "<tr><th>Player</th><th>Min</th><th>Touches</th><th>Def Pen</th>"
            "<th>Def 3rd</th><th>Mid 3rd</th><th>Att 3rd</th><th>Att Pen</th></tr>"
            "</thead>"
            f"<tbody>{rows}</tbody>"
            "<tfoot><tr><th>12 Players</th><td></td><td></td><td></td>"
            "<td></td><td></td><td></td></tr></tfoot></table>")


def test_the_possession_tables_zone_touches_merge_into_the_summary_row(store):
    """The heatmap's whole data source: FBref's own coarse pitch-zone
    breakdown, read from a second table for the same players already parsed
    from the summary one, not a second set of phantom players."""
    from bet.ingest.base import IngestResult
    from bet.ingest.fbref import FBrefSource

    page = ("<html><head><title>Bayern Munich vs Wolfsburg Match Report | FBref"
            "</title></head><body>"
            '<span class="venuetime" data-venue-epoch="1724510400"></span>'
            + _player_table("stats_abc12345_summary", "BAY")
            + _possession_table("stats_abc12345_possession", "BAY")
            + _player_table("stats_def67890_summary", "WOL")
            + _possession_table("stats_def67890_possession", "WOL")
            + "</body></html>")

    source = FBrefSource(store, raw_dir="/tmp", delay=0)
    players, stats = source._parse_match_page(
        page, "bundesliga", "2024-25", IngestResult(source="fbref"))

    assert len(stats) == 24                        # still 12 a side, no doubling
    bay_gk = next(s for s in stats if s["team_id"] == "bayern_munich" and s["position"] == "GK")
    # Row 0 (the goalkeeper): min=90, touches=40, def_pen=5, def_3rd=10,
    # mid_3rd=12, att_3rd=8, att_pen=5, in the possession table's own order.
    assert bay_gk["touches_def_pen"] == 5
    assert bay_gk["touches_def_third"] == 10
    assert bay_gk["touches_mid_third"] == 12
    assert bay_gk["touches_att_third"] == 8
    assert bay_gk["touches_att_pen"] == 5
    # The total from the possession table's own "Touches" column agrees with
    # whichever table supplied it first, rather than the two disagreeing.
    assert bay_gk["touches"] is not None


def _shooting_table(table_id, prefix, count=12):
    """Shot volume plus average distance -- ungrouped on the match-report
    version of this table, matching the bare 'dist' entry in COLUMN_MAP."""
    rows = "".join(
        f'<tr><th><a href="/en/players/{abs(hash(prefix)) % 0xffff:04x}{i:04x}/x">'
        f'{prefix} P{i}</a></th><td>90</td><td>{i % 3}</td><td>{2 + i}</td>'
        f"<td>{1 + i}</td><td>{16.5 + i:.1f}</td></tr>"
        for i in range(count))
    attr = f' id="{table_id}"' if table_id else ""
    return (f"<table{attr}><thead>"
            "<tr><th>Player</th><th>Min</th><th>Gls</th><th>Sh</th>"
            "<th>SoT</th><th>Dist</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _defense_table(table_id, prefix, count=12):
    """Defensive Actions table: Tackles (Tkl, TklW) and Challenges (Att, Lost)."""
    rows = "".join(
        f'<tr><th><a href="/en/players/{abs(hash(prefix)) % 0xffff:04x}{i:04x}/x">'
        f'{prefix} P{i}</a></th><td>90</td><td>{1 + i}</td><td>{i}</td>'
        f"<td>{2 + i}</td><td>{1 if i % 2 else 0}</td></tr>"
        for i in range(count))
    attr = f' id="{table_id}"' if table_id else ""
    return (f"<table{attr}><thead>"
            '<tr><th colspan="2"></th><th colspan="2">Tackles</th>'
            '<th colspan="2">Challenges</th></tr>'
            "<tr><th>Player</th><th>Min</th><th>Tkl</th><th>TklW</th>"
            "<th>Att</th><th>Lost</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _misc_table(table_id, prefix, count=12):
    """Miscellaneous Stats table: Performance (Fld, Recov) and Aerial Duels."""
    rows = "".join(
        f'<tr><th><a href="/en/players/{abs(hash(prefix)) % 0xffff:04x}{i:04x}/x">'
        f'{prefix} P{i}</a></th><td>90</td><td>{i}</td><td>{3 + i}</td>'
        f"<td>{1 + i % 4}</td><td>{i % 2}</td></tr>"
        for i in range(count))
    attr = f' id="{table_id}"' if table_id else ""
    return (f"<table{attr}><thead>"
            '<tr><th colspan="2"></th><th colspan="2">Performance</th>'
            '<th colspan="2">Aerial Duels</th></tr>'
            "<tr><th>Player</th><th>Min</th><th>Fld</th><th>Recov</th>"
            "<th>Won</th><th>Lost</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _goalkeeper_table(table_id, prefix):
    """Just the starting keeper (row 0 of `_player_table`'s own convention)."""
    href = f'/en/players/{abs(hash(prefix)) % 0xffff:04x}0000/x'
    attr = f' id="{table_id}"' if table_id else ""
    return (f"<table{attr}><thead>"
            '<tr><th colspan="2"></th><th colspan="4">Shot Stopping</th></tr>'
            "<tr><th>Player</th><th>Min</th><th>SoTA</th><th>GA</th>"
            "<th>Saves</th><th>Save%</th></tr></thead>"
            f'<tbody><tr><th><a href="{href}">{prefix} P0</a></th>'
            "<td>90</td><td>6</td><td>1</td><td>5</td><td>83.3</td></tr></tbody></table>")


def test_richer_stat_tables_merge_into_the_same_records(store):
    """The next batch of FBref tables -- shooting, defensive actions, misc,
    goalkeeper -- read the same way the possession table already does: by
    content, into records keyed by the player identity already established
    from the summary table, never as a second set of phantom players."""
    from bet.ingest.base import IngestResult
    from bet.ingest.fbref import FBrefSource, YARDS_TO_METRES

    page = ("<html><head><title>Bayern Munich vs Wolfsburg Match Report | FBref"
            "</title></head><body>"
            '<span class="venuetime" data-venue-epoch="1724510400"></span>'
            + _player_table("stats_abc12345_summary", "BAY")
            + _shooting_table("stats_abc12345_shooting", "BAY")
            + _defense_table("stats_abc12345_defense", "BAY")
            + _misc_table("stats_abc12345_misc", "BAY")
            + _goalkeeper_table("keeper_stats_abc12345", "BAY")
            + _player_table("stats_def67890_summary", "WOL")
            + "</body></html>")

    source = FBrefSource(store, raw_dir="/tmp", delay=0)
    players, stats = source._parse_match_page(
        page, "bundesliga", "2024-25", IngestResult(source="fbref"))

    assert len(stats) == 24                        # still 12 a side, no doubling
    bay = [s for s in stats if s["team_id"] == "bayern_munich"]
    gk = next(s for s in bay if s["position"] == "GK")

    # Row 0: Gls=0, Sh=2, SoT=1, Dist=16.5 yards.
    striker = next(s for s in bay if s["avg_shot_distance"] is not None)
    assert striker["avg_shot_distance"] == pytest.approx(16.5 * YARDS_TO_METRES, abs=0.01)

    assert all(s["tackles_won"] is not None for s in bay)
    assert all(s["challenges_attempted"] is not None for s in bay)
    assert all(s["fouls_drawn"] is not None for s in bay)
    assert all(s["recoveries"] is not None for s in bay)
    assert all(s["aerials_won"] is not None for s in bay)

    assert gk["gk_shots_faced"] == 6
    assert gk["gk_goals_against"] == 1
    assert gk["gk_saves"] == 5
    assert gk["gk_save_pct"] == pytest.approx(83.3)

    # The other side never saw these tables -- null, not zero.
    wol = [s for s in stats if s["team_id"] == "vfl_wolfsburg"]
    assert all(s["avg_shot_distance"] is None for s in wol)
    assert all(s["gk_saves"] is None for s in wol)


def test_a_page_without_a_possession_table_leaves_zone_touches_null(store):
    """None must mean 'not parsed', distinguishable from a real zero -- a page
    predating this, or one where the table genuinely is not there."""
    from bet.ingest.base import IngestResult
    from bet.ingest.fbref import FBrefSource

    page = ("<html><head><title>Bayern Munich vs Wolfsburg Match Report | FBref"
            "</title></head><body>"
            '<span class="venuetime" data-venue-epoch="1724510400"></span>'
            + _player_table("stats_abc12345_summary", "BAY")
            + _player_table("stats_def67890_summary", "WOL")
            + "</body></html>")

    source = FBrefSource(store, raw_dir="/tmp", delay=0)
    _, stats = source._parse_match_page(
        page, "bundesliga", "2024-25", IngestResult(source="fbref"))
    assert all(s["touches_def_pen"] is None for s in stats)
