"""Player identity resolution.

The failure that matters here is not a missed player -- squads turn over and a
new signing is normal. It is a *wrong merge*: two players collapsed into one
silently corrupts both their histories and every per-90 rate built on them.
"""

import pytest

from bet.players import (
    make_player_id,
    name_similarity,
    normalise_name,
    player_slug,
    position_group,
    resolve_within_squad,
)


@pytest.mark.parametrize("name,expected", [
    ("Ángel Di María", "angel di maria"),
    ("Kevin De Bruyne", "kevin de bruyne"),
    ("Robert Lewandowski", "robert lewandowski"),
    ("  Harry   Kane  ", "harry kane"),
    ("Neuer, Manuel", "neuer manuel"),
])
def test_names_are_normalised_consistently(name, expected):
    assert normalise_name(name) == expected


def test_source_id_is_preferred_over_a_name_slug():
    """FBref ids are stable; names are not."""
    assert make_player_id("fbref", "d70ce98e") == "fbref:d70ce98e"
    assert make_player_id(name="Harry Kane") == "harry_kane"


def test_make_player_id_requires_something_to_work_with():
    with pytest.raises(ValueError):
        make_player_id()


@pytest.mark.parametrize("short,full", [
    ("H. Kane", "Harry Kane"),
    ("Kane", "Harry Kane"),
    ("J. Musiala", "Jamal Musiala"),
    ("Musiala", "Jamal Musiala"),
    ("Kimmich", "Joshua Kimmich"),
])
def test_abbreviated_names_match_their_full_form(short, full):
    assert name_similarity(short, full) > 0.85


@pytest.mark.parametrize("a,b", [
    ("Harry Kane", "Harvey Elliott"),
    ("Joshua Kimmich", "Jamal Musiala"),
    ("Manuel Neuer", "Marc-Andre ter Stegen"),
])
def test_different_players_do_not_match(a, b):
    assert name_similarity(a, b) < 0.85


def test_squad_resolution_finds_the_right_player():
    squad = {"p1": "Harry Kane", "p2": "Jamal Musiala", "p3": "Joshua Kimmich"}
    assert resolve_within_squad("H. Kane", squad) == "p1"
    assert resolve_within_squad("Musiala", squad) == "p2"


def test_unknown_player_returns_none_rather_than_a_wrong_match():
    """A missing row costs one observation; a wrong match corrupts two players."""
    squad = {"p1": "Harry Kane", "p2": "Jamal Musiala"}
    assert resolve_within_squad("Cristiano Ronaldo", squad) is None


def test_ambiguous_match_is_refused():
    """Two brothers in one squad must not be resolved by coin flip."""
    squad = {"p1": "Lucas Hernandez", "p2": "Theo Hernandez"}
    assert resolve_within_squad("Hernandez", squad) is None


def test_empty_inputs_are_handled():
    assert resolve_within_squad("", {"p1": "Harry Kane"}) is None
    assert resolve_within_squad("Harry Kane", {}) is None
    assert normalise_name("") == ""


@pytest.mark.parametrize("position,group", [
    ("GK", "goalkeeper"), ("DF", "defender"), ("MF", "midfielder"), ("FW", "forward"),
    ("MF,FW", "midfielder"),          # FBref writes the primary position first
    ("CB", "defender"), ("LWB", "defender"), ("AM", "midfielder"), ("ST", "forward"),
    (None, "unknown"), ("???", "unknown"),
])
def test_position_groups(position, group):
    assert position_group(position) == group


def test_slug_is_stable_across_spellings():
    assert player_slug("Ángel Di María") == player_slug("Angel Di Maria")
