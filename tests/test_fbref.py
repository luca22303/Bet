"""FBref parsing.

The two things that break every FBref parser: tables hidden inside HTML
comments, and two-level headers whose group prefix is needed to stop 'Att' from
passing colliding with 'Att' from take-ons.
"""

import pandas as pd
import pytest

from bet.ingest.fbref import (
    COLUMN_MAP,
    extract_player_id,
    flatten_columns,
    parse_minutes,
    strip_html_comments,
)


def test_commented_tables_are_unwrapped():
    """All but the first table on a match page is commented out."""
    html = '<div><!--<table id="stats_abc12345_summary"><tr><td>x</td></tr></table>--></div>'
    unwrapped = strip_html_comments(html)
    assert "<!--" not in unwrapped
    assert 'id="stats_abc12345_summary"' in unwrapped


def test_multilevel_headers_keep_their_group_prefix():
    """Without the prefix, passing 'Att' and take-on 'Att' collide silently."""
    frame = pd.DataFrame([[1, 2, 3]], columns=pd.MultiIndex.from_tuples([
        ("Performance", "Gls"),
        ("Passes", "Att"),
        ("Take-Ons", "Att"),
    ]))
    columns = list(flatten_columns(frame).columns)
    assert columns == ["performance_gls", "passes_att", "take_ons_att"]
    assert len(set(columns)) == 3


def test_ungrouped_columns_drop_the_unnamed_placeholder():
    frame = pd.DataFrame([[1]], columns=pd.MultiIndex.from_tuples([
        ("Unnamed: 0_level_0", "Player")]))
    assert list(flatten_columns(frame).columns) == ["player"]


def test_flat_headers_pass_through():
    frame = pd.DataFrame([[1]], columns=["Player Name"])
    assert list(flatten_columns(frame).columns) == ["player_name"]


@pytest.mark.parametrize("raw,expected", [
    ("90", 90.0), ("45+45", 90.0), ("67", 67.0), ("", 0.0), (None, 0.0),
    ("0", 0.0), ("not a number", 0.0),
])
def test_minutes_parsing(raw, expected):
    assert parse_minutes(raw) == expected


def test_stable_player_id_is_taken_from_the_row_link():
    row = '<tr><td><a href="/en/players/d70ce98e/Harry-Kane">Harry Kane</a></td></tr>'
    assert extract_player_id(row, "Harry Kane") == "fbref:d70ce98e"


def test_player_id_falls_back_to_a_name_slug():
    assert extract_player_id("<tr><td>Harry Kane</td></tr>", "Harry Kane") == "harry_kane"
    assert extract_player_id(None, "Ángel Di María") == "angel_di_maria"


def test_column_map_targets_are_real_schema_columns():
    from bet.ingest.fbref import NUMERIC_COLUMNS
    for target in COLUMN_MAP.values():
        assert target in NUMERIC_COLUMNS, target


def test_touch_zone_headers_flatten_and_map_to_real_columns():
    """FBref's own pitch-zone breakdown of touches, from the possession table.

    'Touches' repeats as both the group header and the total column's own
    label -- the case `flatten_columns` collapses to the bare name -- so only
    the five zone columns need their own entry in COLUMN_MAP.
    """
    from bet.ingest.fbref import NUMERIC_COLUMNS, TOUCH_ZONE_COLUMNS

    frame = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=pd.MultiIndex.from_tuples([
        ("Touches", "Touches"), ("Touches", "Def Pen"), ("Touches", "Def 3rd"),
        ("Touches", "Mid 3rd"), ("Touches", "Att 3rd"), ("Touches", "Att Pen"),
    ]))
    columns = list(flatten_columns(frame).columns)
    assert columns == ["touches", "touches_def_pen", "touches_def_3rd",
                       "touches_mid_3rd", "touches_att_3rd", "touches_att_pen"]

    for flattened in columns[1:]:                # skip the plain total
        assert flattened in COLUMN_MAP, flattened
        assert COLUMN_MAP[flattened] in TOUCH_ZONE_COLUMNS
        assert COLUMN_MAP[flattened] in NUMERIC_COLUMNS


@pytest.mark.parametrize("groups,expected_columns", [
    ([("", "Dist")], ["dist"]),
    ([("Take-Ons", "Att"), ("Take-Ons", "Succ"), ("Take-Ons", "Tkld")],
     ["take_ons_att", "take_ons_succ", "take_ons_tkld"]),
    ([("Tackles", "TklW"), ("Challenges", "Att"), ("Challenges", "Lost")],
     ["tackles_tklw", "challenges_att", "challenges_lost"]),
    ([("Performance", "Fld"), ("Performance", "Recov")],
     ["performance_fld", "performance_recov"]),
    ([("Aerial Duels", "Won"), ("Aerial Duels", "Lost")],
     ["aerial_duels_won", "aerial_duels_lost"]),
    ([("Shot Stopping", "SoTA"), ("Shot Stopping", "GA"),
      ("Shot Stopping", "Saves"), ("Shot Stopping", "Save%")],
     ["shot_stopping_sota", "shot_stopping_ga",
      "shot_stopping_saves", "shot_stopping_save%"]),
])
def test_richer_stat_headers_flatten_and_map_to_real_columns(groups, expected_columns):
    """The shooting, possession, defensive-actions, misc and goalkeeper tables
    all follow the same 'group prefix disambiguates' convention already
    proven for the touch zones -- this just extends coverage to the next
    batch of tables, so a wrong guess at a header name fails a test rather
    than silently leaving a column null forever."""
    from bet.ingest.fbref import NUMERIC_COLUMNS

    columns_index = pd.MultiIndex.from_tuples(
        [(upper or "Unnamed: 0_level_0", lower) for upper, lower in groups])
    frame = pd.DataFrame([list(range(len(groups)))], columns=columns_index)
    columns = list(flatten_columns(frame).columns)
    assert columns == expected_columns

    for flattened in columns:
        assert flattened in COLUMN_MAP, flattened
        assert COLUMN_MAP[flattened] in NUMERIC_COLUMNS


def test_shot_distance_is_converted_from_yards_to_metres():
    """FBref reports it in yards; the rest of this project's spatial code
    (pitch.py) works in metres, so it must not leak the source unit."""
    from bet.ingest.fbref import FBrefSource, YARDS_TO_METRES

    record = {"avg_shot_distance": None}
    row = pd.Series({"dist": "20.0"})
    FBrefSource._merge_stats(record, ["dist"], row)
    assert record["avg_shot_distance"] == pytest.approx(20.0 * YARDS_TO_METRES)


def test_a_value_already_present_is_not_overwritten_by_a_later_table():
    """First non-null wins, same rule as every other merged column."""
    from bet.ingest.fbref import FBrefSource

    record = {"avg_shot_distance": 15.0}
    row = pd.Series({"dist": "99.0"})
    FBrefSource._merge_stats(record, ["dist"], row)
    assert record["avg_shot_distance"] == 15.0
