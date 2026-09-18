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
