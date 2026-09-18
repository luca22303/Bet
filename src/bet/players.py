"""Canonical player identity.

The same problem as club names but harder, because there is no fixed roster to
check against and the spellings are worse: FBref writes "Ángel Di María",
kicker writes "Di Maria", Understat writes "Angel Di Maria". Left alone, one
player becomes three, each with a third of the minutes, and every per-90 rate
built on top is wrong.

Two mechanisms, in order of reliability:

1.  Source ids. FBref puts a stable id in every player URL
    (/en/players/d70ce98e/Harry-Kane). When a source offers one, it is used and
    nothing is guessed.
2.  Normalised name slug. Accents stripped, punctuation removed, ordering
    preserved. Good enough for most, and `resolve_within_squad` narrows the
    remainder by matching against a known squad, which is what makes surname-only
    sources like a lineup listing usable at all.

Unlike clubs, an unknown player is not an error: squads turn over constantly and
a new signing is a normal event, not a bug. The guard here is different — a
player must never be silently merged with a different one.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Tokens that appear in some spellings of a name and not others.
_NOISE = {"jr", "junior", "sr", "senior", "de", "da", "do", "dos", "das", "van",
          "von", "der", "den", "el", "al", "bin", "ibn"}


def normalise_name(name: str) -> str:
    """Strip accents, punctuation and case from a player name."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name).strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ß", "ss").replace("ø", "o").replace("đ", "d").replace("ł", "l")
    text = re.sub(r"[.'`´’\-_]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def player_slug(name: str) -> str:
    """Canonical id derived from a name.

    Used only when a source offers no stable id of its own. Two different players
    with identical normalised names would collide; that is rare within one
    league, and `resolve_within_squad` is the mitigation where it matters.
    """
    return normalise_name(name).replace(" ", "_")


def make_player_id(source: str | None = None, source_id: str | None = None,
                   name: str | None = None) -> str:
    """Prefer a source's stable id; fall back to the name slug."""
    if source and source_id:
        return f"{source}:{source_id}"
    if name:
        return player_slug(name)
    raise ValueError("need either a source id or a name")


def _tokens(name: str) -> list[str]:
    return [t for t in normalise_name(name).split() if t not in _NOISE]


def name_similarity(left: str, right: str) -> float:
    """Similarity in [0, 1], tolerant of the ways sources abbreviate names.

    Surname agreement carries most of the weight, because that is what short
    forms keep. "H. Kane" and "Harry Kane" should match; "Harry Kane" and
    "Harvey Elliott" should not.
    """
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    if left_tokens == right_tokens:
        return 1.0

    # Surnames are the last token in every source's ordering seen so far.
    surname_score = SequenceMatcher(None, left_tokens[-1], right_tokens[-1]).ratio()

    # An initial matching a full first name is a strong signal, not a weak one.
    given_score = 0.0
    if len(left_tokens) > 1 and len(right_tokens) > 1:
        a, b = left_tokens[0], right_tokens[0]
        if a == b:
            given_score = 1.0
        elif len(a) == 1 or len(b) == 1:
            given_score = 1.0 if a[0] == b[0] else 0.0
        else:
            given_score = SequenceMatcher(None, a, b).ratio()
    elif len(left_tokens) == 1 or len(right_tokens) == 1:
        # One source gave a surname only; judge on the surname alone.
        return surname_score

    return 0.75 * surname_score + 0.25 * given_score


def resolve_within_squad(name: str, squad: dict[str, str], *,
                         threshold: float = 0.85) -> str | None:
    """Match a loose name against a known squad.

    `squad` maps player_id -> full name. Returns the best match above the
    threshold, or None. Returning None is the safe outcome: a missing player
    costs one row, while a wrong match silently pollutes two players' histories.

    An ambiguous match — two candidates within a whisker of each other — also
    returns None rather than picking one.
    """
    if not name or not squad:
        return None

    scored = sorted(
        ((name_similarity(name, full_name), player_id)
         for player_id, full_name in squad.items()),
        reverse=True,
    )
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
        return None      # too close to call
    return scored[0][1]


# Position groups. Props and availability weighting need the group, not the
# exact label, and every source spells the exact label differently.
_POSITION_GROUPS = {
    "GK": "goalkeeper",
    "DF": "defender", "CB": "defender", "LB": "defender", "RB": "defender",
    "WB": "defender", "LWB": "defender", "RWB": "defender",
    "MF": "midfielder", "DM": "midfielder", "CM": "midfielder", "AM": "midfielder",
    "LM": "midfielder", "RM": "midfielder",
    "FW": "forward", "ST": "forward", "CF": "forward", "LW": "forward", "RW": "forward",
}


def position_group(position: str | None) -> str:
    """Collapse a source's position label onto one of four groups."""
    if not position:
        return "unknown"
    # FBref writes multi-position players as "MF,FW"; the first is the primary.
    primary = str(position).split(",")[0].strip().upper()
    if primary in _POSITION_GROUPS:
        return _POSITION_GROUPS[primary]
    for prefix, group in _POSITION_GROUPS.items():
        if primary.startswith(prefix):
            return group
    return "unknown"
