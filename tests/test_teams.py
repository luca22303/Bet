import pytest

from bet.teams import UnknownTeamError, display_name, known_team_ids, resolve


@pytest.mark.parametrize("name", [
    "Bayern Munich", "Bayern", "FC Bayern München", "fc bayern munchen", "BAYERN MUNICH",
])
def test_bayern_aliases_collapse(name):
    assert resolve(name) == "bayern_munich"


@pytest.mark.parametrize("name", [
    "M'gladbach", "Borussia Mönchengladbach", "Gladbach", "Borussia Monchengladbach",
])
def test_gladbach_aliases_collapse(name):
    assert resolve(name) == "borussia_monchengladbach"


def test_cross_source_spellings_agree():
    # football-data.co.uk, ClubElo, OpenLigaDB and Understat spellings.
    assert resolve("Ein Frankfurt") == resolve("Frankfurt") == resolve("Eintracht Frankfurt")
    assert resolve("FC Koln") == resolve("Cologne") == resolve("1. FC Köln")


def test_unknown_team_raises_rather_than_passing_through():
    # A pipeline that silently invents a team is worse than one that stops.
    with pytest.raises(UnknownTeamError):
        resolve("Manchester United")


def test_unknown_team_can_be_skipped_explicitly():
    assert resolve("Manchester United", strict=False) is None


def test_every_team_id_has_a_display_name():
    for team_id in known_team_ids():
        assert display_name(team_id)


def test_founding_years_are_stripped_as_a_class():
    """Enumerating years breaks whenever a club with an unlisted one is promoted.

    `1. FC Heidenheim 1846` is the name OpenLigaDB serves, and 1846 was not on
    the list, so it did not resolve at all.
    """
    assert resolve("1. FC Heidenheim 1846") == "1_fc_heidenheim"
    assert resolve("VfL Bochum 1848") == "vfl_bochum"
    assert resolve("TSG 1899 Hoffenheim") == "tsg_hoffenheim"


def test_a_year_that_identifies_a_club_survives_the_strip():
    """1860 is the club's name, not a founding-year suffix.

    The year pass runs last precisely so this resolves on its own alias first;
    stripping digits eagerly would leave "munchen" and lose the club.
    """
    assert resolve("TSV 1860 München") == "1860_munich"
    assert resolve("FC Bayern München") == "bayern_munich"


def test_german_listing_abbreviations_expand():
    """"Bor." and "Ein." are how German sources shorten club names."""
    assert resolve("Bor. Mönchengladbach") == "borussia_monchengladbach"
    assert resolve("Bor. Dortmund") == "borussia_dortmund"
    assert resolve("Ein. Frankfurt") == "eintracht_frankfurt"


def test_no_two_clubs_collide_under_the_relaxed_matching():
    """Every registered name must still resolve to its own club.

    Each fallback pass widens what matches, and a pass that is too eager maps
    two clubs onto one id -- which would silently merge their match histories
    rather than raise.
    """
    from bet.teams import _TEAMS

    for team_id, (display, aliases) in _TEAMS.items():
        for name in (display, team_id.replace("_", " "), *aliases):
            assert resolve(name, strict=False) == team_id, f"{name!r} -> wrong club"
