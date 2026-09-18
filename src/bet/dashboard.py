"""A static HTML dashboard, generated from the store.

Deliberately a single self-contained file rather than a server. There is nothing
to deploy, nothing to keep running, and the output can be opened from disk or
mailed to someone. Regenerate it whenever the data changes.

It shows what the system knows and, just as importantly, what it does not: data
coverage per source, whether each XI is confirmed or predicted, how much
evidence sits behind every prop, and whether market prices were even available
to compare against. A dashboard that only shows conclusions invites more
confidence than the numbers deserve.
"""

from __future__ import annotations

import html
from datetime import datetime, timedelta

import pandas as pd

from bet.config import GERMAN_STAKE_TAX, OUTCOMES

# Light and dark are both defined so the page follows the reader's system
# setting rather than assuming one.
STYLE = """
:root {
  --bg: #fbfbfa; --panel: #ffffff; --ink: #1a1a18; --muted: #6b6b66;
  --line: #e4e4e0; --accent: #2f6f4e; --warn: #9a5b12; --bad: #9a2b2b;
  --home: #2f6f4e; --draw: #8a8a84; --away: #3a5a8f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14140f; --panel: #1c1c18; --ink: #f0f0ea; --muted: #9a9a92;
    --line: #2e2e28; --accent: #6bbf90; --warn: #d9a441; --bad: #e07a7a;
    --home: #6bbf90; --draw: #9a9a92; --away: #7fa3d9;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 32px 0 12px; font-weight: 600; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.cards { display: grid; gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; }
.card.value { border-color: var(--accent); }
.teams { font-weight: 600; font-size: 16px; display: flex; justify-content: space-between;
  align-items: baseline; gap: 12px; flex-wrap: wrap; }
.when { color: var(--muted); font-size: 12px; font-weight: 400; }
.bar { display: flex; height: 22px; border-radius: 5px; overflow: hidden;
  margin: 10px 0 6px; background: var(--line); }
.seg { display: flex; align-items: center; justify-content: center; font-size: 11px;
  color: #fff; font-variant-numeric: tabular-nums; min-width: 0; }
.seg.h { background: var(--home); } .seg.d { background: var(--draw); }
.seg.a { background: var(--away); }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px;
  color: var(--muted); font-variant-numeric: tabular-nums; }
.meta b { color: var(--ink); font-weight: 600; }
.tag { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 20px;
  border: 1px solid var(--line); color: var(--muted); }
.tag.ok { color: var(--accent); border-color: var(--accent); }
.tag.warn { color: var(--warn); border-color: var(--warn); }
.ev { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line);
  font-size: 13px; }
.ev .pos { color: var(--accent); font-weight: 600; }
.note { font-size: 12px; color: var(--warn); margin-top: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 13px;
  font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.05em; }
td.num, th.num { text-align: right; }
.panel { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 4px 4px 0; overflow-x: auto; }
.warnbox { background: var(--panel); border: 1px solid var(--warn);
  border-radius: 10px; padding: 12px 16px; font-size: 13px; }
.warnbox li { margin: 3px 0; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px;
  border-top: 1px solid var(--line); padding-top: 14px; }
"""


def _e(value) -> str:
    return html.escape(str(value))


def _team(team_id: str) -> str:
    """Readable club name, falling back to the id for anything unmapped."""
    from bet.teams import display_name
    try:
        return display_name(team_id)
    except KeyError:
        return team_id.replace("_", " ").title()


def _probability_bar(probabilities: dict[str, float]) -> str:
    segments = []
    for outcome, css in (("H", "h"), ("D", "d"), ("A", "a")):
        share = probabilities.get(outcome, 0.0)
        # Below about 7% a label does not fit, so the segment is drawn bare
        # rather than clipping text.
        label = f"{share:.0%}" if share > 0.07 else ""
        segments.append(
            f'<div class="seg {css}" style="width:{share * 100:.2f}%">{label}</div>')
    return f'<div class="bar">{"".join(segments)}</div>'


def _match_card(match) -> str:
    classes = "card value" if match.value_bets else "card"
    parts = [
        f'<div class="{classes}">',
        '<div class="teams"><span>'
        f'{_e(_team(match.home_team))} <span class="when">vs</span> '
        f'{_e(_team(match.away_team))}</span>'
        f'<span class="when">{match.kickoff:%a %d %b %H:%M}</span></div>',
        _probability_bar(match.probabilities),
    ]

    meta = [f'expected goals <b>{match.expected_home_goals:.2f} - '
            f'{match.expected_away_goals:.2f}</b>']
    meta.append("fair " + " / ".join(
        f'<b>{match.fair_odds[o]:.2f}</b>' for o in OUTCOMES))
    if match.market_odds:
        meta.append("market " + " / ".join(
            f"{match.market_odds[o]:.2f}" for o in OUTCOMES))
    parts.append(f'<div class="meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>')

    # Whether the XI is known or guessed changes how much any of this is worth.
    lineup_bits = []
    for side in ("home", "away"):
        lineup = match.lineups.get(side)
        if lineup is None or not getattr(lineup, "starters", None):
            continue
        label = _team(match.home_team if side == "home" else match.away_team)
        if lineup.is_confirmed:
            tag = '<span class="tag ok">confirmed</span>'
        else:
            tag = f'<span class="tag">predicted {lineup.confidence:.0%}</span>'
        effect = match.absence_effect.get(side, 1.0)
        strength = "" if abs(effect - 1) < 0.01 else f" &middot; attack x{effect:.2f}"
        lineup_bits.append(f"<span>{_e(label)}: <b>{_e(lineup.formation)}</b> {tag}{strength}</span>")
    if lineup_bits:
        parts.append(f'<div class="meta" style="margin-top:6px">{"".join(lineup_bits)}</div>')

    for side in ("home", "away"):
        absent = match.absences.get(side, [])
        if absent:
            names = ", ".join(_e(a.split("_")[-1]) for a in absent[:6])
            parts.append(f'<div class="meta"><span>out ({side}): {names}</span></div>')

    if match.value_bets:
        rows = []
        for bet in match.value_bets:
            label = {"H": _team(match.home_team), "D": "Draw",
                     "A": _team(match.away_team)}[bet["selection"]]
            rows.append(
                f'<div><span class="pos">VALUE</span> {_e(label)} @ {bet["odds"]:.2f} '
                f'&middot; EV <span class="pos">{bet["ev"]:+.1%}</span> '
                f'&middot; stake {bet["stake"]:.2%} of bankroll '
                f'&middot; model {bet["model_probability"]:.1%} vs market '
                f'{bet["market_probability"]:.1%}</div>')
        parts.append(f'<div class="ev">{"".join(rows)}</div>')
    elif match.market_odds:
        parts.append('<div class="ev" style="color:var(--muted)">no value at current prices</div>')

    for note in match.notes:
        parts.append(f'<div class="note">{_e(note)}</div>')

    parts.append("</div>")
    return "".join(parts)


def _table(frame: pd.DataFrame, numeric: set[str] | None = None) -> str:
    if frame.empty:
        return '<div class="panel" style="padding:14px;color:var(--muted)">no data</div>'
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{_e(c)}</th>' if c in numeric else f"<th>{_e(c)}</th>"
        for c in frame.columns)
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row):
            if isinstance(value, float):
                text = f"{value:.3f}"
            else:
                text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
            cells.append(f'<td class="num">{_e(text)}</td>' if column in numeric
                         else f"<td>{_e(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<div class="panel"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def render(store, brief, *, coverage: pd.DataFrame | None = None,
           quality_issues: list | None = None) -> str:
    """Build the whole page."""
    generated = datetime.utcnow()

    sections = [
        "<h1>Bundesliga model</h1>",
        f'<div class="sub">Generated {generated:%Y-%m-%d %H:%M} UTC &middot; '
        f"{len(brief.matches)} fixture(s) in the next {brief.window_days} days &middot; "
        f"{brief.value_bet_count} value bet(s) at current prices</div>",
    ]

    if brief.warnings:
        items = "".join(f"<li>{_e(w)}</li>" for w in brief.warnings)
        sections.append(f'<div class="warnbox"><b>Data gaps</b><ul>{items}</ul></div>')

    sections.append("<h2>Fixtures</h2>")
    if brief.matches:
        sections.append(
            f'<div class="cards">{"".join(_match_card(m) for m in brief.matches)}</div>')
    else:
        sections.append('<div class="panel" style="padding:14px;color:var(--muted)">'
                        "no fixtures in the window</div>")

    if not brief.prop_picks.empty:
        sections.append("<h2>Player props &mdash; highest expected counts</h2>")
        props = brief.prop_picks.copy()
        keep = [c for c in ("player_id", "team_id", "opponent_id", "line",
                            "expected_minutes", "expected", "p_over", "fair_over",
                            "sample_90s") if c in props.columns]
        props = props[keep].rename(columns={
            "player_id": "player", "team_id": "team", "opponent_id": "opponent",
            "expected_minutes": "minutes", "expected": "exp", "p_over": "P(over)",
            "fair_over": "fair", "sample_90s": "90s"})
        sections.append(_table(props, numeric={"line", "minutes", "exp",
                                               "P(over)", "fair", "90s"}))
        sections.append('<div class="sub" style="margin-top:8px">'
                        "A low <b>90s</b> means the rate is mostly prior, not evidence.</div>")

    if coverage is not None and not coverage.empty:
        sections.append("<h2>Data coverage</h2>")
        sections.append(_table(coverage, numeric=set(coverage.columns) - {"season"}))

    if quality_issues:
        items = "".join(f"<li>{_e(str(i))}</li>" for i in quality_issues[:12])
        sections.append("<h2>Data quality</h2>")
        sections.append(f'<div class="panel" style="padding:12px 16px"><ul>{items}</ul></div>')

    sections.append(
        "<footer>Fair odds and expected value are net of a "
        f"{GERMAN_STAKE_TAX:.1%} betting-stake tax and bookmaker margin. "
        "A model edge is not a proven edge &mdash; check <code>bet backtest</code> "
        "before acting on anything here. Probabilities come from a time-weighted "
        "Dixon-Coles fit priced off the expected eleven.</footer>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bundesliga model</title><style>{STYLE}</style></head>
<body><div class="wrap">{"".join(sections)}</div></body></html>"""


def build(store, as_of: datetime | None = None, *, days: int = 8,
          league: str = "bundesliga", include_quality: bool = True) -> str:
    """Compile a brief and render it."""
    from bet.recommend import build_brief

    as_of = as_of or datetime.utcnow()
    brief = build_brief(store, as_of, days=days, league=league)

    coverage, issues = None, None
    if include_quality:
        from bet.quality import run_quality_checks
        report = run_quality_checks(store, as_of)
        coverage = report.coverage
        issues = report.errors + report.warnings

    return render(store, brief, coverage=coverage, quality_issues=issues)
