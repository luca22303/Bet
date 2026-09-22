"""Canonical team identities and cross-source name resolution.

Every source spells clubs differently: football-data.co.uk says "Bayern Munich",
ClubElo says "Bayern", OpenLigaDB says "FC Bayern Munchen". If those are allowed
to reach the store as three different teams the model silently trains on
fragments of a squad, so all ingestion funnels through `resolve`.

An unknown name raises rather than being passed through. A pipeline that
quietly invents a team is worse than one that stops.
"""

from __future__ import annotations

import re
import unicodedata

# canonical_id -> (display name, alias spellings seen in the wild)
_TEAMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "bayern_munich": ("Bayern Munich", ("bayern", "fc bayern munchen", "fc bayern munich", "bayern munchen", "fc bayern")),
    "borussia_dortmund": ("Borussia Dortmund", ("dortmund", "bvb", "borussia dortmund", "bv borussia 09 dortmund")),
    "rb_leipzig": ("RB Leipzig", ("leipzig", "rb leipzig", "rasenballsport leipzig")),
    "bayer_leverkusen": ("Bayer Leverkusen", ("leverkusen", "bayer 04 leverkusen", "bayer leverkusen")),
    "borussia_monchengladbach": ("Borussia Monchengladbach", ("mgladbach", "m'gladbach", "gladbach", "monchengladbach", "borussia monchengladbach", "borussia mgladbach", "vfl borussia monchengladbach")),
    "eintracht_frankfurt": ("Eintracht Frankfurt", ("ein frankfurt", "frankfurt", "eintracht frankfurt")),
    "vfl_wolfsburg": ("VfL Wolfsburg", ("wolfsburg", "vfl wolfsburg")),
    "sc_freiburg": ("SC Freiburg", ("freiburg", "sc freiburg")),
    "tsg_hoffenheim": ("TSG Hoffenheim", ("hoffenheim", "tsg 1899 hoffenheim", "tsg hoffenheim", "1899 hoffenheim")),
    "fc_union_berlin": ("Union Berlin", ("union berlin", "fc union berlin", "1 fc union berlin", "union")),
    "vfb_stuttgart": ("VfB Stuttgart", ("stuttgart", "vfb stuttgart")),
    "sv_werder_bremen": ("Werder Bremen", ("werder bremen", "bremen", "sv werder bremen")),
    "fsv_mainz_05": ("Mainz 05", ("mainz", "mainz 05", "1 fsv mainz 05", "fsv mainz 05")),
    "fc_augsburg": ("FC Augsburg", ("augsburg", "fc augsburg")),
    "vfl_bochum": ("VfL Bochum", ("bochum", "vfl bochum", "vfl bochum 1848")),
    "1_fc_koln": ("1. FC Koln", ("fc koln", "koln", "cologne", "1 fc koln", "1. fc koln")),
    "fc_st_pauli": ("FC St. Pauli", ("st pauli", "fc st pauli", "st. pauli")),
    "holstein_kiel": ("Holstein Kiel", ("holstein kiel", "kiel")),
    "hamburger_sv": ("Hamburger SV", ("hamburg", "hamburger sv", "hsv")),
    "1_fc_heidenheim": ("1. FC Heidenheim", ("heidenheim", "1 fc heidenheim", "fc heidenheim", "1. fc heidenheim")),
    "sv_darmstadt_98": ("SV Darmstadt 98", ("darmstadt", "sv darmstadt 98")),
    "hertha_bsc": ("Hertha BSC", ("hertha", "hertha bsc", "hertha berlin")),
    "schalke_04": ("Schalke 04", ("schalke 04", "schalke", "fc schalke 04")),
    "arminia_bielefeld": ("Arminia Bielefeld", ("bielefeld", "arminia bielefeld", "dsc arminia bielefeld")),
    "greuther_furth": ("Greuther Furth", ("greuther furth", "furth", "spvgg greuther furth")),
    "fc_nurnberg": ("1. FC Nurnberg", ("nurnberg", "1 fc nurnberg", "nuremberg")),
    "fortuna_dusseldorf": ("Fortuna Dusseldorf", ("fortuna dusseldorf", "dusseldorf")),
    "hannover_96": ("Hannover 96", ("hannover", "hannover 96")),
    "sc_paderborn": ("SC Paderborn 07", ("paderborn", "sc paderborn 07", "sc paderborn")),

    # Bundesliga clubs from earlier in the backfill window. A 2010-onward
    # ingest hits Kaiserslautern, Braunschweig and Ingolstadt, and an unknown
    # name raises rather than passing through -- so leaving them out turns a
    # historical backfill into a wall of errors.
    "1_fc_kaiserslautern": ("1. FC Kaiserslautern", ("kaiserslautern", "1 fc kaiserslautern", "fck")),
    "eintracht_braunschweig": ("Eintracht Braunschweig", ("braunschweig", "eintracht braunschweig")),
    "fc_ingolstadt": ("FC Ingolstadt 04", ("ingolstadt", "fc ingolstadt 04", "fc ingolstadt")),
    "vfl_wolfsburg_ii": ("VfL Wolfsburg II", ("wolfsburg ii",)),
    "energie_cottbus": ("Energie Cottbus", ("cottbus", "energie cottbus")),
    "vfl_osnabruck": ("VfL Osnabruck", ("osnabruck", "vfl osnabruck")),
    "alemannia_aachen": ("Alemannia Aachen", ("aachen", "alemannia aachen")),

    # Second-tier clubs. Needed for the bundesliga2 league code, and for the
    # promoted-team prior, which reads a side's record from the division it
    # came up from.
    "hansa_rostock": ("Hansa Rostock", ("hansa rostock", "rostock")),
    "msv_duisburg": ("MSV Duisburg", ("duisburg", "msv duisburg")),
    "karlsruher_sc": ("Karlsruher SC", ("karlsruhe", "karlsruher sc", "ksc")),
    "sv_sandhausen": ("SV Sandhausen", ("sandhausen", "sv sandhausen")),
    "jahn_regensburg": ("Jahn Regensburg", ("jahn regensburg", "regensburg")),
    "sv_elversberg": ("SV Elversberg", ("elversberg", "sv elversberg")),
    "1_fc_magdeburg": ("1. FC Magdeburg", ("magdeburg", "1 fc magdeburg")),
    "sv_wehen_wiesbaden": ("SV Wehen Wiesbaden", ("wehen", "wehen wiesbaden", "sv wehen")),
    "ssv_ulm": ("SSV Ulm 1846", ("ulm", "ssv ulm", "ssv ulm 1846")),
    "eintracht_trier": ("Eintracht Trier", ("trier",)),
    "preussen_munster": ("Preussen Munster", ("munster", "preussen munster")),
    "spvgg_unterhaching": ("SpVgg Unterhaching", ("unterhaching",)),
    "fc_erzgebirge_aue": ("Erzgebirge Aue", ("aue", "erzgebirge aue")),
    "fsv_frankfurt": ("FSV Frankfurt", ("fsv frankfurt",)),
    "vfr_aalen": ("VfR Aalen", ("aalen", "vfr aalen")),
    "sc_verl": ("SC Verl", ("verl", "sc verl")),
    "sv_darmstadt_ii": ("SV Darmstadt II", ("darmstadt ii",)),
    "würzburger_kickers": ("Wurzburger Kickers", ("wurzburger kickers", "wurzburg")),
    "dynamo_dresden": ("Dynamo Dresden", ("dynamo dresden", "dresden")),
    "1860_munich": ("1860 Munich", ("1860 munich", "munich 1860", "tsv 1860 munchen", "1860 munchen")),
    "kickers_offenbach": ("Kickers Offenbach", ("offenbach", "kickers offenbach")),
    "rot_weiss_essen": ("Rot-Weiss Essen", ("rot weiss essen", "essen")),
    "sc_freiburg_ii": ("SC Freiburg II", ("freiburg ii",)),
    "hallescher_fc": ("Hallescher FC", ("halle", "hallescher fc")),
    "fc_saarbrucken": ("1. FC Saarbrucken", ("saarbrucken", "1 fc saarbrucken")),
    "viktoria_koln": ("Viktoria Koln", ("viktoria koln",)),
    "waldhof_mannheim": ("Waldhof Mannheim", ("waldhof mannheim", "mannheim")),
    "tsv_havelse": ("TSV Havelse", ("havelse",)),
    "fc_homburg": ("FC 08 Homburg", ("homburg",)),
    "stuttgarter_kickers": ("Stuttgarter Kickers", ("stuttgarter kickers",)),
    "sportfreunde_lotte": ("Sportfreunde Lotte", ("lotte",)),
    "chemnitzer_fc": ("Chemnitzer FC", ("chemnitz", "chemnitzer fc")),
    "fc_carl_zeiss_jena": ("Carl Zeiss Jena", ("jena", "carl zeiss jena")),
}


def _normalise(name: str) -> str:
    """Strip accents, punctuation and common corporate noise from a club name."""
    text = unicodedata.normalize("NFKD", name.strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ß", "ss").replace("&", " and ")
    text = re.sub(r"[.'`\-_/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_LOOKUP: dict[str, str] = {}
for _cid, (_display, _aliases) in _TEAMS.items():
    _LOOKUP[_normalise(_display)] = _cid
    _LOOKUP[_normalise(_cid.replace("_", " "))] = _cid
    for _alias in _aliases:
        _LOOKUP[_normalise(_alias)] = _cid


class UnknownTeamError(KeyError):
    """Raised when a source name cannot be mapped to a canonical club."""


def resolve(name: str, *, strict: bool = True) -> str | None:
    """Map a source-specific club name onto a canonical team id."""
    if name is None:
        if strict:
            raise UnknownTeamError("cannot resolve a null team name")
        return None

    key = _normalise(name)
    if key in _LOOKUP:
        return _LOOKUP[key]

    # Second pass: drop legal-form and sponsor tokens, then retry.
    stripped = " ".join(
        token
        for token in key.split()
        if token not in {"fc", "sv", "vfl", "vfb", "sc", "tsg", "fsv", "dsc", "spvgg", "bv", "1", "04", "05", "98", "07", "96", "1899", "1848", "09"}
    ).strip()
    if stripped and stripped in _LOOKUP:
        return _LOOKUP[stripped]

    if strict:
        raise UnknownTeamError(
            f"unknown club {name!r} (normalised {key!r}); add it to bet.teams._TEAMS"
        )
    return None


def display_name(team_id: str) -> str:
    return _TEAMS[team_id][0]


def known_team_ids() -> tuple[str, ...]:
    return tuple(_TEAMS)
