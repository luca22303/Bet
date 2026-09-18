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
