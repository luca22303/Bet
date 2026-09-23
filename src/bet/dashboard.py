"""A static HTML dashboard, generated from the store.

One self-contained file rather than a server: nothing to deploy, nothing to keep
running, and it opens from disk. Regenerate it whenever the data changes.

Four views. An overview of the round, a fixture list where any match opens into
its own detail -- formation on a pitch, both squads with their stats, the likely
scorelines -- then model health and player props.

The design commitment is that it shows what the system does not know beside what
it does: whether an XI is confirmed or predicted and with what confidence, how
much evidence sits behind a player's rate, whether market prices existed to
compare against at all. A dashboard of conclusions alone invites more confidence
than the numbers deserve.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bet.config import GERMAN_STAKE_TAX, OUTCOMES
from bet.viz import (
    Series,
    diverging_bar,
    esc,
    hbar_chart,
    line_chart,
    pitch,
    scatter,
    stat_tile,
)

STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --home: #2a78d6; --draw: #d3d2cc; --away: #e34948;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  --turf: #f2f4f0; --pitchline: #d7dbd4;
  --seq-100: #cde2fb; --seq-450: #2a78d6;
}
* { box-sizing: border-box; }
body { margin:0; padding:0 0 64px; background:var(--page); color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1120px; margin:0 auto; padding:0 16px; }
header { padding:26px 0 14px; }
h1 { font-size:25px; margin:0 0 3px; letter-spacing:-0.015em; }
.sub { color:var(--ink-2); font-size:13px; }
h2 { font-size:12px; text-transform:uppercase; letter-spacing:0.09em;
  color:var(--muted); margin:30px 0 11px; font-weight:600; }
h3 { font-size:14px; margin:0 0 8px; font-weight:600; }

nav { display:flex; gap:2px; border-bottom:1px solid var(--border); margin-top:14px;
  overflow-x:auto; }
nav button { background:none; border:none; border-bottom:2px solid transparent;
  padding:9px 14px; font:inherit; font-size:14px; color:var(--ink-2);
  cursor:pointer; white-space:nowrap; }
nav button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--s1);
  font-weight:600; }
.view { display:none; } .view.on { display:block; }

.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px; margin-top:16px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:13px 15px; }
.tilevalue { font-size:27px; font-weight:600; letter-spacing:-0.02em; line-height:1.15; }
.tilelabel { font-size:12px; color:var(--ink-2); margin-top:2px; }
.tiledetail { font-size:11.5px; color:var(--muted); margin-top:4px; }
.badge { display:inline-flex; align-items:center; gap:3px; font-size:10.5px;
  padding:1px 6px; border-radius:20px; border:1px solid currentColor; margin-left:3px; }
.badge.good{color:var(--good)} .badge.warning{color:var(--warning)}
.badge.critical{color:var(--critical)}

.card { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  margin-bottom:9px; overflow:hidden; }
.card.value { border-color:var(--s3); }
.card.correct { border-color:var(--good); border-width:1px 1px 1px 3px; }
.card.miss { border-color:var(--critical); border-width:1px 1px 1px 3px; }
.cardhead { display:grid; grid-template-columns:1fr auto; gap:10px; padding:13px 15px;
  cursor:pointer; align-items:center; }
.cardhead:hover { background:color-mix(in srgb, var(--ink) 4%, transparent); }
.cardhead:focus-visible { outline:2px solid var(--s1); outline-offset:-2px; }
.fixture { font-weight:600; font-size:15.5px; }
.kick { color:var(--muted); font-size:12px; font-weight:400; }
.chev { color:var(--muted); font-size:12px; transition:transform .15s; }
.card[data-open="1"] .chev { transform:rotate(90deg); }

.dbar { display:flex; border-radius:5px; overflow:hidden; margin:9px 0 5px;
  background:var(--grid); gap:2px; }
.dbar .seg { display:flex; align-items:center; justify-content:center;
  font-size:11px; font-weight:600; color:#fff; min-width:0;
  font-variant-numeric:tabular-nums; }
.seg-home{background:var(--home)} .seg-away{background:var(--away)}
.seg-draw{background:var(--draw); color:var(--ink)}
.legend { display:flex; gap:14px; font-size:11.5px; color:var(--ink-2); margin-top:5px;
  flex-wrap:wrap; }
.key { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:4px;
  vertical-align:baseline; }

.meta { display:flex; flex-wrap:wrap; gap:5px 17px; font-size:12.5px;
  color:var(--ink-2); font-variant-numeric:tabular-nums; }
.meta b { color:var(--ink); font-weight:600; }
.tag { display:inline-block; font-size:10.5px; padding:1px 7px; border-radius:20px;
  border:1px solid var(--border); color:var(--muted); }
.tag.ok { color:var(--good); border-color:var(--good); }
.tag.miss { color:var(--critical); border-color:var(--critical); }
.ev { margin-top:9px; padding-top:9px; border-top:1px solid var(--border); font-size:13px; }
.pos { color:var(--good); font-weight:600; }
.neg { color:var(--critical); font-weight:600; }
.note { font-size:12px; color:var(--warning); margin-top:5px; }

.detail { display:none; padding:0 15px 15px; border-top:1px solid var(--border); }
.card[data-open="1"] .detail { display:block; }

.subtabs { margin-top:2px; }
.subnav { display:flex; gap:2px; border-bottom:1px solid var(--border); margin-bottom:12px; }
.subnav button { background:none; border:none; border-bottom:2px solid transparent;
  padding:7px 12px; font:inherit; font-size:13px; color:var(--ink-2); cursor:pointer; }
.subnav button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--s1);
  font-weight:600; }
.subview { display:none; }
.subview.on { display:block; }
.playermark { cursor:pointer; }
.playermark:focus-visible circle.player { outline:2px solid var(--s1); outline-offset:2px; }

#player-modal-backdrop { position:fixed; inset:0; background:rgba(11,11,11,0.45);
  display:flex; align-items:center; justify-content:center; z-index:100; padding:16px; }
#player-modal-backdrop[hidden] { display:none; }
#player-modal { background:var(--surface); border-radius:12px; padding:20px 22px;
  max-width:360px; width:100%; max-height:80vh; overflow-y:auto; position:relative;
  box-shadow:0 12px 40px rgba(11,11,11,0.25); }
#pm-close { position:absolute; top:10px; right:12px; background:none; border:none;
  font-size:22px; line-height:1; color:var(--muted); cursor:pointer; padding:4px; }
#pm-close:hover { color:var(--ink); }
#player-modal h3 { font-size:17px; margin:0 0 2px; padding-right:20px; }
#pm-rows > div { display:flex; justify-content:space-between; gap:14px;
  padding:5px 0; border-bottom:1px solid var(--border); font-size:13.5px; }
#pm-rows > div:last-child { border-bottom:none; }
#pm-rows b { font-weight:600; }

.history { margin-bottom:14px; }
.history summary { cursor:pointer; list-style:none; font-size:13px;
  font-weight:600; color:var(--ink-2); padding:9px 2px;
  display:flex; align-items:center; gap:6px; }
.history summary::-webkit-details-marker { display:none; }
.history summary::before { content:"\\25B8"; font-size:11px; color:var(--muted);
  transition:transform .15s; }
.history[open] summary::before { transform:rotate(90deg); }
.history summary:hover { color:var(--ink); }
.historybody { padding-top:4px; }
.pitches { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:13px 0; }
@media (max-width:640px){ .pitches{grid-template-columns:1fr} .tiles{grid-template-columns:1fr 1fr} }
.pitchwrap { text-align:center; }
.pitchwrap h3 { font-size:13px; margin-bottom:5px; }
.pitch { width:100%; max-width:300px; height:auto; }
.turf { fill:var(--turf); }
.pitchline { stroke:var(--pitchline); stroke-width:1.5; fill:none; }
.player { fill:var(--s1); stroke:var(--surface); stroke-width:2; }
.player.uncertain { fill:none; stroke:var(--s1); stroke-width:2; stroke-dasharray:3 2.5; }
.pnum { font-size:8.5px; font-weight:700; fill:#fff; }
.player.uncertain + .pnum { fill:var(--ink-2); }
.pname { font-size:8.5px; fill:var(--ink-2); }

table { width:100%; border-collapse:collapse; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:5px 9px; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:600; font-size:10.5px; text-transform:uppercase;
  letter-spacing:0.05em; }
td.num,th.num { text-align:right; }
tr.out td { color:var(--muted); text-decoration:line-through; }
.panel { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:13px 15px; overflow-x:auto; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width:760px){ .grid2{grid-template-columns:1fr} }

.chart { width:100%; height:auto; display:block; }
.grid { stroke:var(--grid); stroke-width:1; }
.grid.dashed { stroke-dasharray:4 4; }
.axis { stroke:var(--axis); stroke-width:1; }
.tick,.axislabel,.rowlabel,.ptlabel,.quad,.reflabel { fill:var(--muted); font-size:10.5px; }
.rowlabel { fill:var(--ink-2); font-size:12px; }
.rowlabel.strong { fill:var(--ink); font-weight:600; }
.barvalue { fill:var(--ink); font-size:11.5px; font-variant-numeric:tabular-nums; }
.benchlabel { fill:var(--muted); font-size:10px; }
.serieslabel { font-size:11px; font-weight:600; }
.s1-ink{fill:var(--s1)} .s2-ink{fill:var(--s2)} .s3-ink{fill:var(--s3)}
.line { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
.line.s1{stroke:var(--s1)} .line.s2{stroke:var(--s2)} .line.s3{stroke:var(--s3)}
.dot { stroke:var(--surface); stroke-width:2; }
.dot.s1{fill:var(--s1)} .dot.s2{fill:var(--s2)} .dot.s3{fill:var(--s3)}
.reference { fill:none; stroke:var(--axis); stroke-width:1.5; stroke-dasharray:5 4; }
.bar { fill:var(--s1); }
.bar.benchmark { fill:var(--s1); opacity:0.45; }
.pt { fill:var(--s1); stroke:var(--surface); stroke-width:2; }
.caption { font-size:11px; color:var(--muted); margin-top:5px; }
.empty { color:var(--muted); font-size:12.5px; padding:14px; }
.warnbox { background:var(--surface); border:1px solid var(--warning); border-radius:10px;
  padding:11px 15px; font-size:12.5px; margin-top:14px; }
.warnbox ul { margin:5px 0 0; padding-left:18px; }
footer { margin-top:38px; color:var(--muted); font-size:12px;
  border-top:1px solid var(--border); padding-top:14px; }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
  background:var(--ink); color:var(--page); font-size:11.5px; padding:4px 8px;
  border-radius:5px; z-index:50; white-space:nowrap; }
.controls { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  margin-top:14px; padding:10px 14px; border:1px solid var(--border);
  border-radius:10px; background:var(--surface); font-size:13px; }
.btn { font:inherit; font-size:13px; padding:5px 13px; border-radius:7px;
  border:1px solid var(--s1); background:var(--s1); color:#fff; cursor:pointer; }
.btn:disabled { opacity:.55; cursor:default; }
.btn.secondary { background:none; color:var(--s1); }
#refresh-status { color:var(--ink-2); }
#refresh-status.err { color:var(--critical); }
/* Not var(--warning): #fab219 is a badge fill, and as text on
   white it falls well under 4.5:1. This is the same hue, dark
   enough to read. */
#refresh-status.warn { color:#8a5a00; }
#refresh-status .errlist { margin:6px 0 0; padding-left:18px; font-size:12px; line-height:1.5; max-height:9em; overflow-y:auto; }
#refresh-status .errlist li { margin:2px 0; word-break:break-word; }
.spin { display:inline-block; width:11px; height:11px; margin-right:6px;
  border:2px solid var(--border); border-top-color:var(--s1); border-radius:50%;
  animation:spin .8s linear infinite; vertical-align:-1px; }
@keyframes spin { to { transform:rotate(360deg); } }
.provenance { border-radius:10px; padding:11px 15px; font-size:12.5px;
  margin:14px 0 0; border:1px solid var(--border); background:var(--surface); }
.provenance.synthetic { border-color:var(--critical); }
.provenance.stale { border-color:var(--warning); }
.provenance b { display:block; margin-bottom:2px; }
.stale-cell { color:var(--warning); }
"""

SCRIPT = """
(function(){
  var tip = document.getElementById('tip');
  document.addEventListener('pointerover', function(e){
    var t = e.target.closest('[data-tip]');
    if(!t){ tip.style.opacity = 0; return; }
    tip.innerHTML = t.getAttribute('data-tip');
    tip.style.opacity = 1;
    var r = t.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8,
                              Math.max(8, r.left + r.width/2 - tip.offsetWidth/2)) + 'px';
    tip.style.top = Math.max(8, r.top - tip.offsetHeight - 8) + 'px';
  });
  document.addEventListener('pointerleave', function(){ tip.style.opacity = 0; }, true);

  document.querySelectorAll('nav button').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('nav button').forEach(function(x){
        x.setAttribute('aria-selected', x === b); });
      document.querySelectorAll('.view').forEach(function(v){
        v.classList.toggle('on', v.id === b.dataset.view); });
    });
  });

  // Formation/Heatmaps/Ticker inside one fixture's detail. Scoped to the
  // nearest .subtabs so every card's own switcher is independent -- there is
  // one of these per fixture on the page, not one globally.
  document.querySelectorAll('.subnav button').forEach(function(b){
    b.addEventListener('click', function(e){
      e.stopPropagation();
      var group = b.closest('.subtabs');
      group.querySelectorAll('.subnav button').forEach(function(x){
        x.setAttribute('aria-selected', x === b); });
      group.querySelectorAll('.subview').forEach(function(v){
        v.classList.toggle('on', v.id === b.dataset.subview); });
    });
  });

  function toggle(card){
    card.dataset.open = card.dataset.open === '1' ? '0' : '1';
    card.querySelector('.cardhead').setAttribute(
      'aria-expanded', card.dataset.open === '1');
  }
  document.querySelectorAll('.cardhead').forEach(function(h){
    h.addEventListener('click', function(){ toggle(h.parentElement); });
    h.addEventListener('keydown', function(e){
      if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); toggle(h.parentElement); }
    });
  });

  // Clicking a player's mark on the pitch opens their stats for this fixture.
  // The payload is embedded on the mark itself (data-player, a JSON blob) so
  // this needs no lookup table and works for every pitch on the page, current
  // matchday or previous.
  var backdrop = document.getElementById('player-modal-backdrop');
  var pmName = document.getElementById('pm-name');
  var pmTeam = document.getElementById('pm-team');
  var pmRows = document.getElementById('pm-rows');
  var pmNote = document.getElementById('pm-note');

  function openPlayerModal(payload){
    var data;
    try { data = JSON.parse(payload); } catch(e){ return; }
    pmName.textContent = data.name || '';
    pmTeam.textContent = data.team || '';
    pmRows.innerHTML = (data.rows || []).map(function(row){
      return '<div><span>' + row[0] + '</span><b>' + row[1] + '</b></div>';
    }).join('');
    pmNote.textContent = data.note || '';
    pmNote.style.display = data.note ? '' : 'none';
    backdrop.hidden = false;
    document.getElementById('pm-close').focus();
  }

  function closePlayerModal(){ backdrop.hidden = true; }

  document.addEventListener('click', function(e){
    var mark = e.target.closest('.playermark');
    if(mark){ e.stopPropagation(); openPlayerModal(mark.getAttribute('data-player')); return; }
    if(e.target === backdrop || e.target.id === 'pm-close'){ closePlayerModal(); }
  });
  document.addEventListener('keydown', function(e){
    var mark = e.target.closest && e.target.closest('.playermark');
    if(mark && (e.key === 'Enter' || e.key === ' ')){
      e.preventDefault(); e.stopPropagation();
      openPlayerModal(mark.getAttribute('data-player'));
      return;
    }
    if(e.key === 'Escape' && !backdrop.hidden){ closePlayerModal(); }
  });

})();
"""


def _team(team_id: str) -> str:
    from bet.teams import display_name
    try:
        return display_name(team_id)
    except KeyError:
        return str(team_id).replace("_", " ").title()


def _short(player_id: str) -> str:
    return str(player_id).split("_")[-1].upper()


def _player_name(player_id: str, names: dict[str, str]) -> str:
    name = names.get(player_id)
    if name:
        return name.split()[-1] if " " in name else name
    return str(player_id).split("_")[-1].replace("-", " ").title()


def _num(value, digits: int = 0) -> str | None:
    """A stat as a plain string for the modal, or `None` to hide the row.

    `None` rather than "0" or "0.0" for a value that was never recorded (goals
    on an upcoming fixture, say) -- a real zero and an absent number look
    identical as text, and only one of them is a fact.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return f"{value:.{digits}f}"


def _player_modal(player_id: str, name: str, team: str, group: str, position: str,
                  propensity: float | None, days_since_start,
                  box_score: dict | None, played: bool) -> dict:
    """What clicking this player's marker on the pitch shows.

    A played match has its own box score -- the real minutes, goals and
    assists from this fixture. An upcoming one only has the rolling per-90
    rate that fed the prediction, and says so rather than presenting a rate
    as if it were this match's number.
    """
    rows: list[tuple[str, str]] = [("Position", position or group.replace("_", " ").title())]

    if played and box_score:
        rows.append(("Minutes", _num(box_score.get("minutes")) or "0"))
        rows.append(("Goals", _num(box_score.get("goals")) or "0"))
        rows.append(("Assists", _num(box_score.get("assists")) or "0"))
        if group != "goalkeeper":
            shots = _num(box_score.get("shots"))
            if shots is not None:
                rows.append(("Shots", shots))
            sot = _num(box_score.get("shots_on_target"))
            if sot is not None:
                rows.append(("Shots on target", sot))
            xg = _num(box_score.get("xg"), 2)
            if xg is not None:
                rows.append(("xG", xg))
            tackles = _num(box_score.get("tackles"))
            if tackles is not None:
                rows.append(("Tackles", tackles))
            interceptions = _num(box_score.get("interceptions"))
            if interceptions is not None:
                rows.append(("Interceptions", interceptions))
        yellow = box_score.get("yellow_cards") or 0
        red = box_score.get("red_cards") or 0
        if yellow or red:
            rows.append(("Cards", f"{int(yellow)} yellow" + (f", {int(red)} red" if red else "")))
        note = ("No goalkeeper-specific stats (saves, goals conceded) yet -- "
                "FBref publishes them in a separate table this parser does not "
                "read." if group == "goalkeeper" else "")
    elif played:
        rows.append(("Minutes", "0"))
        note = "Named in the squad but did not play, per the source ingested."
    else:
        rows.append(("Start confidence",
                     f"{propensity:.0%}" if propensity is not None else "&mdash;"))
        rows.append(("Last started",
                     f"{int(days_since_start)}d ago" if days_since_start is not None
                     and not pd.isna(days_since_start) else "never"))
        note = ("Not played yet -- these are the rolling per-90 rates behind the "
                "prediction, not this match's own numbers.")

    return {"name": name, "team": team, "rows": rows, "note": note}


# ----------------------------------------------------------- expected score


# A scoreline is shown only when it is genuinely the mode and clearly ahead of
# the runner-up. Football scorelines are flat: in an even match 1-1, 1-0 and 0-1
# often sit within a point of each other, and printing whichever happens to lead
# would dress a coin-flip as a prediction. Blank is the honest output.
MIN_SCORE_PROBABILITY = 0.09
MIN_SCORE_MARGIN = 1.20


def most_likely_score(matrix: np.ndarray, *,
                      min_probability: float = MIN_SCORE_PROBABILITY,
                      min_margin: float = MIN_SCORE_MARGIN) -> dict:
    """The predicted scoreline, or nothing when the distribution is too flat.

    Returns `{score, probability, runner_up, margin, confident, reason}`.
    `score` is None when the model cannot separate the leading scorelines, and
    the caller renders a blank rather than a number.
    """
    flat = [((h, a), float(matrix[h, a]))
            for h in range(matrix.shape[0]) for a in range(matrix.shape[1])]
    flat.sort(key=lambda item: item[1], reverse=True)
    if not flat:
        return {"score": None, "probability": 0.0, "runner_up": None,
                "margin": 0.0, "confident": False, "reason": "no distribution"}

    (top_h, top_a), top_p = flat[0]
    (second_h, second_a), second_p = flat[1] if len(flat) > 1 else ((0, 0), 0.0)
    margin = top_p / second_p if second_p > 0 else float("inf")

    if top_p < min_probability:
        reason = f"most likely scoreline only {top_p:.0%} — too flat to call"
        confident = False
    elif margin < min_margin:
        reason = (f"{top_h}-{top_a} and {second_h}-{second_a} are within "
                  f"{(margin - 1) * 100:.0f}% of each other")
        confident = False
    else:
        reason = ""
        confident = True

    return {
        "score": f"{top_h}-{top_a}" if confident else None,
        "probability": top_p,
        "runner_up": f"{second_h}-{second_a}",
        "runner_up_probability": second_p,
        "margin": margin,
        "confident": confident,
        "reason": reason,
    }


def top_scorelines(matrix: np.ndarray, count: int = 6) -> list[tuple[str, float]]:
    flat = [(f"{h}-{a}", float(matrix[h, a]))
            for h in range(matrix.shape[0]) for a in range(matrix.shape[1])]
    flat.sort(key=lambda item: item[1], reverse=True)
    return flat[:count]


# -------------------------------------------------------------- match detail


def build_match_detail(store, model, match, as_of: datetime,
                       rates: pd.DataFrame, names: dict[str, str],
                       prop_model=None) -> dict:
    """Everything the expanded view of one fixture needs."""
    from bet.lineups import start_propensity
    from bet.models.dixon_coles import score_matrix

    matrix = score_matrix(match.expected_home_goals, match.expected_away_goals,
                          model.params.rho)

    detail = {
        "matrix": matrix,
        "score": most_likely_score(matrix),
        "scorelines": top_scorelines(matrix),
        "squads": {},
    }

    # A played match has its own box score -- goals, assists, minutes actually
    # played -- which is what clicking a player should show first. An
    # upcoming one has only the rolling per-90 rate, which is context for a
    # guess rather than a record, and the modal says which one it is looking
    # at rather than presenting a rate as though it were this match's number.
    played = match.actual_outcome is not None
    box_score: dict[str, dict] = {}
    if played:
        all_ids = list(match.lineups.get("home").starters if match.lineups.get("home") else []) + \
                  list(match.lineups.get("home").bench if match.lineups.get("home") else []) + \
                  list(match.lineups.get("away").starters if match.lineups.get("away") else []) + \
                  list(match.lineups.get("away").bench if match.lineups.get("away") else [])
        if all_ids:
            actual = store.player_stats_as_of(as_of, player_ids=all_ids)
            if not actual.empty:
                actual = actual[actual["match_id"] == match.match_id]
                box_score = {row.player_id: row._asdict()
                            for row in actual.itertuples(index=False)}

    for side, team_id in (("home", match.home_team), ("away", match.away_team)):
        lineup = match.lineups.get(side)
        if lineup is None:
            continue

        absent = set(match.absences.get(side, []))
        # Recency per player, so the table can show who is actually playing
        # rather than only who has a rate on file.
        propensity_table = start_propensity(store, team_id, as_of)
        recency = (dict(zip(propensity_table["player_id"],
                            propensity_table["days_since_start"]))
                   if not propensity_table.empty else {})
        players = []
        for player_id in lineup.starters:
            row = rates[rates["player_id"] == player_id]
            stats = row.iloc[0].to_dict() if not row.empty else {}
            from bet.players import position_group
            group = position_group(stats.get("position"))
            days_since_start = (None if pd.isna(recency.get(player_id))
                                else recency.get(player_id))
            players.append({
                "player_id": player_id,
                "name": _player_name(player_id, names),
                "short": _short(player_id),
                "group": group,
                "position": stats.get("position") or "",
                "propensity": lineup.propensities.get(player_id),
                "minutes": float(stats.get("minutes") or 0.0),
                "xg90": float(stats.get("xg_p90") or 0.0),
                "shots90": float(stats.get("shots_p90") or 0.0),
                "tackles90": float(stats.get("tackles_p90") or 0.0),
                "days_since_start": days_since_start,
                "modal": _player_modal(
                    player_id, _player_name(player_id, names), _team(team_id),
                    group, stats.get("position") or "", lineup.propensities.get(player_id),
                    days_since_start, box_score.get(player_id), played),
            })

        bench = []
        for player_id in (lineup.bench or [])[:7]:
            row = rates[rates["player_id"] == player_id]
            stats = row.iloc[0].to_dict() if not row.empty else {}
            bench.append({
                "player_id": player_id,
                "name": _player_name(player_id, names),
                "propensity": lineup.propensities.get(player_id),
                "minutes": float(stats.get("minutes") or 0.0),
                "out": player_id in absent,
            })

        detail["squads"][side] = {
            "team_id": team_id,
            "lineup": lineup,
            "players": players,
            "bench": bench,
            "absent": sorted(absent),
        }

    return detail


# A player who has not started in this long is shown in the warning colour.
# Roughly three matchdays: long enough to survive a rested weekend, short enough
# that a genuinely absent player stands out.
STALE_PLAYER_DAYS = 28


def _player_table(players: list[dict]) -> str:
    """The eleven, with how recently each of them actually played.

    The `last` column exists because a per-90 rate carries no date. A striker
    who has not featured since September still has a rate, and without this
    column he sits in the table looking exactly like someone who played on
    Saturday.
    """
    rows = []
    for player in players:
        propensity = player.get("propensity")
        start = f"{propensity:.0%}" if propensity is not None else "&mdash;"

        days = player.get("days_since_start")
        # The column arrives as a float because it carries NaN for a player who
        # has never started; "0.0d" reads like a rounding artefact rather than a
        # date, so it is shown as a whole number of days.
        if days is None or (isinstance(days, float) and pd.isna(days)):
            last, last_cls = "never", ' class="num stale-cell"'
        elif int(days) > STALE_PLAYER_DAYS:
            last, last_cls = f"{int(days)}d", ' class="num stale-cell"'
        else:
            last, last_cls = f"{int(days)}d", ' class="num"'

        rows.append(
            f"<tr><td>{esc(player['name'])}</td>"
            f"<td>{esc(player['position'] or player['group'][:3].upper())}</td>"
            f'<td class="num">{start}</td>'
            f"<td{last_cls}>{last}</td>"
            f'<td class="num">{player["xg90"]:.2f}</td>'
            f'<td class="num">{player["shots90"]:.2f}</td>'
            f'<td class="num">{player["tackles90"]:.2f}</td>'
            f'<td class="num">{player["minutes"]:.0f}</td></tr>')
    return (
        '<table><thead><tr><th>player</th><th>pos</th>'
        '<th class="num">starts</th><th class="num">last</th>'
        '<th class="num">xG/90</th>'
        '<th class="num">sh/90</th><th class="num">tkl/90</th>'
        '<th class="num">mins</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>")


def _bench_list(bench: list[dict]) -> str:
    if not bench:
        return ""
    items = []
    for player in bench:
        cls = ' class="out"' if player["out"] else ""
        propensity = player.get("propensity")
        share = f" {propensity:.0%}" if propensity else ""
        items.append(f"<span{cls}>{esc(player['name'])}"
                     f'<span class="kick">{share}</span></span>')
    return ('<div class="meta" style="margin-top:7px"><span class="kick">bench:</span> '
            + " &middot; ".join(items) + "</div>")


# -------------------------------------------------------------------- views


def _outcome_class(match) -> str:
    """`card correct`/`card miss` once a result is known, else `""`.

    Shared so a fixture that finishes while still shown as part of the
    current matchday gets exactly the same treatment as one already moved
    into the previous-matchday panel -- the same fact, the same colour,
    wherever the card happens to live.
    """
    if match.predicted_correct is True:
        return "card correct"
    if match.predicted_correct is False:
        return "card miss"
    return ""


def _verdict_tag(match) -> str:
    """The text form of the same verdict -- colour is never the only signal."""
    if match.predicted_correct is True:
        return '<span class="tag ok">correct</span>'
    if match.predicted_correct is False:
        return '<span class="tag miss">missed</span>'
    return ""


def _probability_block(match, labels: dict) -> list[str]:
    """The diverging bar, legend and fair/market odds line.

    Shared by the live fixture card and the history card: both are "here is
    what the model thought", and the only thing that differs between them is
    what happened after -- nothing, or a final score.
    """
    body = [diverging_bar(match.probabilities, labels)]
    body.append(
        '<div class="legend">'
        f'<span><i class="key" style="background:var(--home)"></i>{esc(labels["H"])} '
        f'{match.probabilities["H"]:.0%}</span>'
        f'<span><i class="key" style="background:var(--draw)"></i>Draw '
        f'{match.probabilities["D"]:.0%}</span>'
        f'<span><i class="key" style="background:var(--away)"></i>{esc(labels["A"])} '
        f'{match.probabilities["A"]:.0%}</span></div>')

    meta = ["fair " + " / ".join(f"<b>{match.fair_odds[o]:.2f}</b>" for o in OUTCOMES)]
    if match.market_odds:
        meta.append("market " + " / ".join(f"{match.market_odds[o]:.2f}" for o in OUTCOMES))
    body.append(f'<div class="meta" style="margin-top:7px">'
                + "".join(f"<span>{m}</span>" for m in meta) + "</div>")
    return body


def _fixture_card(match, detail: dict | None, index: int) -> str:
    """A fixture in the current matchday.

    Most of these are still to be played and carry no verdict. One that has
    already kicked off and finished -- the round is a weekend, and not every
    match in it shares a kick-off time -- gets the same green/red treatment
    as a card in the previous-matchday panel, for the same reason: the result
    is known, so the card should say whether the model called it.
    """
    labels = {"H": _team(match.home_team), "D": "Draw", "A": _team(match.away_team)}
    outcome_class = _outcome_class(match)
    # A settled result outranks the value-bet accent -- whether the pick was
    # right is the more important fact once it is knowable, and the value bet
    # itself is still shown (and marked won/lost) in the body below.
    classes = outcome_class or ("card value" if match.value_bets else "card")

    if match.actual_outcome is not None:
        score_html = (f'<span class="fixture">{match.actual_home_goals:.0f} '
                     f'&ndash; {match.actual_away_goals:.0f}</span>')
        head_tip = "full time"
    else:
        score = (detail or {}).get("score") or {}
        if score.get("score"):
            score_html = (f'<span class="fixture" data-tip="most likely scoreline &middot; '
                        f'{score["probability"]:.1%}">{esc(score["score"])}</span>')
        else:
            # Deliberately blank: the model cannot separate the leading
            # scorelines, and a number here would dress a coin-flip as a
            # prediction.
            score_html = ('<span class="kick" data-tip="'
                        + esc(score.get("reason", "no scoreline prediction"))
                        + '">&mdash;</span>')
        head_tip = None

    verdict = _verdict_tag(match)
    kick_line = (f'{match.kickoff:%a %d %b %H:%M}' + (f" &middot; {head_tip}" if head_tip else ""))

    head = (
        f'<div class="cardhead" role="button" tabindex="0" aria-expanded="false">'
        f'<div><div class="fixture">{esc(_team(match.home_team))} '
        f'<span class="kick">vs</span> {esc(_team(match.away_team))}</div>'
        f'<div class="kick">{kick_line}</div></div>'
        f'<div style="text-align:right">{score_html} {verdict}'
        f'<div class="kick">{"predicted " if match.actual_outcome is not None else ""}'
        f'{match.expected_home_goals:.2f} &ndash; '
        f'{match.expected_away_goals:.2f} xG <span class="chev">&#9656;</span></div>'
        f"</div></div>")

    body = _probability_block(match, labels)

    if match.value_bets:
        rows = []
        for bet in match.value_bets:
            rows.append(f'<div><span class="pos">VALUE</span> {esc(labels[bet["selection"]])} '
                        f'@ {bet["odds"]:.2f} &middot; EV <span class="pos">{bet["ev"]:+.1%}</span> '
                        f'&middot; stake {bet["stake"]:.2%} of bankroll</div>')
        body.append(f'<div class="ev">{"".join(rows)}</div>')
    elif match.market_odds:
        body.append('<div class="ev" style="color:var(--muted)">no value at current prices</div>')

    for note in match.notes:
        body.append(f'<div class="note">{esc(note)}</div>')

    return (f'<div class="{classes}" data-open="0">{head}'
            f'<div style="padding:0 15px 13px">{"".join(body)}</div>'
            f"{_match_detail_html(match, detail, index)}</div>")


def _history_card(match, detail: dict | None, index: int) -> str:
    """A played fixture: the point-in-time prediction, and what happened.

    Fit strictly before that matchday kicked off (see
    `bet.recommend.previous_matchday_brief`), so this is what the model would
    genuinely have said beforehand -- not a prediction dressed up with
    hindsight -- shown next to the result it is being judged against.

    Opens into the same formation/lineup/scoreline detail a live fixture card
    does, when there is any: whether fbref/kicker ever had player data for
    this match does not depend on it being in the past.
    """
    labels = {"H": _team(match.home_team), "D": "Draw", "A": _team(match.away_team)}
    verdict = _verdict_tag(match)

    actual = (f'{match.actual_home_goals:.0f} &ndash; {match.actual_away_goals:.0f}'
             if match.actual_outcome is not None else "&mdash;")

    head = (
        f'<div class="cardhead" role="button" tabindex="0" aria-expanded="false">'
        f'<div><div class="fixture">{esc(_team(match.home_team))} '
        f'<span class="kick">vs</span> {esc(_team(match.away_team))}</div>'
        f'<div class="kick">{match.kickoff:%a %d %b %H:%M} &middot; full time</div></div>'
        f'<div style="text-align:right"><span class="fixture">{actual}</span> {verdict}'
        f'<div class="kick">predicted {match.expected_home_goals:.2f} &ndash; '
        f'{match.expected_away_goals:.2f} xG <span class="chev">&#9656;</span></div>'
        "</div></div>")

    body = _probability_block(match, labels)

    if match.value_bets:
        rows = []
        for bet in match.value_bets:
            won = bet["selection"] == match.actual_outcome
            tag = '<span class="pos">WON</span>' if won else '<span class="neg">LOST</span>'
            rows.append(f'<div>{tag} {esc(labels[bet["selection"]])} '
                        f'@ {bet["odds"]:.2f} &middot; EV would have been '
                        f'<span class="pos">{bet["ev"]:+.1%}</span></div>')
        body.append(f'<div class="ev">{"".join(rows)}</div>')
    elif match.market_odds:
        body.append('<div class="ev" style="color:var(--muted)">'
                    "no value at the prices knowable beforehand</div>")

    for note in match.notes:
        body.append(f'<div class="note">{esc(note)}</div>')

    classes = _outcome_class(match) or "card"
    return (f'<div class="{classes}" data-open="0">{head}'
            f'<div style="padding:0 15px 13px">{"".join(body)}</div>'
            f"{_match_detail_html(match, detail, index)}</div>")


def _match_detail_html(match, detail: dict | None, index: int) -> str:
    if not detail or not detail.get("squads"):
        return ('<div class="detail"><div class="empty">No line-up data. '
                "Run <code>bet ingest --source fbref</code> to populate squads."
                "</div></div>")

    parts = ['<div class="detail">']

    # A team with no appearance history at all -- fbref (or kicker for the
    # confirmed XI) was never ingested -- gets `DEFAULT_FORMATION` at 0%
    # confidence and an empty starters list. That is the model correctly
    # saying "nothing is known", not a real prediction, and drawing an empty
    # pitch under a formation label dressed it up as one. Distinguishing it
    # from a genuine low-confidence guess (which still has starters, just an
    # uncertain XI, and is worth showing with its dashed-ring markers) matters:
    # one is "no data", the other is "some data, honestly uncertain".
    no_data_sides = [side for side in ("home", "away")
                     if detail["squads"].get(side)
                     and not detail["squads"][side]["players"]]

    # No early return here even when neither side has any player data at
    # all: the loop below already renders a plain "no data" box per side in
    # that case, and Scorelines and the model's own numbers below need no
    # player data whatsoever -- they come from the match model, not fbref.
    # Returning early used to throw those away too, so a fixture with no
    # lineups showed literally nothing on the page instead of what the model
    # actually does know about it.
    uid = match.match_id.replace(":", "-")

    pitches = []
    for side in ("home", "away"):
        squad = detail["squads"].get(side)
        if not squad:
            continue
        if side in no_data_sides:
            pitches.append(
                f'<div class="pitchwrap"><h3>{esc(_team(squad["team_id"]))}</h3>'
                '<div class="empty" style="margin-top:20px">No player data for this '
                "club yet.</div></div>")
            continue
        lineup = squad["lineup"]
        source = ('<span class="tag ok">confirmed</span>' if lineup.is_confirmed
                  else f'<span class="tag">predicted {lineup.confidence:.0%}</span>')
        players = [{**p, "tip": f'{esc(p["name"])} &middot; {p["position"] or p["group"]}'
                    + (f' &middot; starts {p["propensity"]:.0%}'
                       if p.get("propensity") is not None else "")}
                   for p in squad["players"]]
        pitches.append(
            f'<div class="pitchwrap"><h3>{esc(_team(squad["team_id"]))}</h3>'
            f'<div class="meta" style="justify-content:center"><span>'
            f'<b>{esc(lineup.formation)}</b> {source}</span></div>'
            + pitch(lineup.formation, players, team_name=_team(squad["team_id"]),
                    confirmed=lineup.is_confirmed)
            + "".join(f'<div class="note">{esc(note)}</div>' for note in lineup.notes)
            + "</div>")

    formation_tab = [f'<div class="pitches">{"".join(pitches)}</div>']

    if no_data_sides:
        names = " and ".join(esc(_team(detail["squads"][s]["team_id"])) for s in no_data_sides)
        formation_tab.append(f'<div class="caption">No player data for {names} yet -- run '
                             "<code>bet ingest --source fbref</code> to predict its line-up "
                             "too.</div>")

    if not any(detail["squads"][s]["lineup"].is_confirmed
               for s in detail["squads"]):
        formation_tab.append('<div class="caption">A dashed ring marks a player the model '
                             "is less than 55% sure will start. Positions show the named "
                             "shape, not tracked movement.</div>")

    heatmap_tab = (
        '<div class="empty">Heatmaps need touch-location data no source ingested '
        "here currently provides. A coarse zone-based version (FBref's six pitch "
        "zones per player) is planned; a smooth touch heatmap like Sofascore's "
        "needs its own, unverified source.</div>")

    ticker_tab = (
        '<div class="empty">No live play-by-play source is ingested yet. Planned '
        "from kicker's match ticker, written against its page structure and "
        "verified once reachable from outside this sandbox -- like the confirmed "
        "line-up scraper already is.</div>")

    parts.append(
        f'<div class="subtabs"><nav class="subnav" role="tablist">'
        f'<button data-subview="{uid}-formation" aria-selected="true">Formation</button>'
        f'<button data-subview="{uid}-heatmaps" aria-selected="false">Heatmaps</button>'
        f'<button data-subview="{uid}-ticker" aria-selected="false">Ticker</button>'
        f'</nav>'
        f'<div class="subview on" id="{uid}-formation">{"".join(formation_tab)}</div>'
        f'<div class="subview" id="{uid}-heatmaps">{heatmap_tab}</div>'
        f'<div class="subview" id="{uid}-ticker">{ticker_tab}</div>'
        f'</div>')

    # Likely scorelines, as a small ranked chart rather than a table of numbers.
    score = detail["score"]
    scorelines = detail["scorelines"]
    heading = (f"Most likely: <b>{esc(score['score'])}</b> at "
               f"{score['probability']:.1%}" if score["score"]
               else f'No scoreline called &mdash; {esc(score["reason"])}')
    parts.append(
        f'<div class="grid2" style="margin-top:6px">'
        f'<div><h3>Scorelines</h3><div class="meta" style="margin-bottom:6px">'
        f"<span>{heading}</span></div>"
        + hbar_chart([(name, p) for name, p in scorelines], width=420,
                     value_format="{:.1%}", lower_is_better=False)
        + "</div>")

    squad_tables = []
    for side in ("home", "away"):
        squad = detail["squads"].get(side)
        if not squad or side in no_data_sides:
            continue        # already explained above; an empty table repeats it
        absent = (f'<div class="meta" style="margin-top:6px"><span class="kick">'
                  f'unavailable: {esc(", ".join(_short(a) for a in squad["absent"]))}'
                  f"</span></div>") if squad["absent"] else ""
        squad_tables.append(
            f'<h3 style="margin-top:12px">{esc(_team(squad["team_id"]))}</h3>'
            + _player_table(squad["players"]) + _bench_list(squad["bench"]) + absent
            + '<div class="caption">Per-90 rates use the last 18 months, shrunk '
              "toward the squad mean by minutes played. <b>last</b> is days since "
              "that player actually started &mdash; a rate carries no date, so "
              "without it a player who stopped playing in September looks "
              "identical to one who played on Saturday.</div>")
    parts.append(f'<div>{"".join(squad_tables)}</div></div>')

    parts.append("</div>")
    return "".join(parts)


def _overview(brief, coverage, quality_report) -> str:
    confirmed = sum(
        1 for m in brief.matches for side in ("home", "away")
        if m.lineups.get(side) is not None and m.lineups[side].is_confirmed)
    total_sides = max(len(brief.matches) * 2, 1)

    priced = sum(1 for m in brief.matches if m.market_odds)
    errors = len(quality_report.errors) if quality_report else 0
    warnings = len(quality_report.warnings) if quality_report else 0
    status = "critical" if errors else ("warning" if warnings else "good")

    tiles = [
        stat_tile(str(len(brief.matches)), "fixtures",
                  detail=brief.scope_label),
        stat_tile(str(brief.value_bet_count), "value bets",
                  detail="net of margin and tax" if priced else "no market prices"),
        stat_tile(f"{confirmed}/{total_sides}", "confirmed XIs",
                  detail="the rest are predicted"),
        stat_tile(f"{errors}E {warnings}W", "data quality", status=status,
                  detail="run bet quality for detail"),
    ]

    parts = [f'<div class="tiles">{"".join(tiles)}</div>']

    if brief.warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in brief.warnings)
        parts.append(f'<div class="warnbox"><b>Data gaps</b><ul>{items}</ul></div>')

    if coverage is not None and not coverage.empty:
        parts.append("<h2>Coverage by season</h2>")
        parts.append('<div class="panel">' + _frame_table(coverage) + "</div>")

    if quality_report and (quality_report.errors or quality_report.warnings):
        parts.append("<h2>Quality checks</h2>")
        items = "".join(f"<li>{esc(str(i))}</li>"
                        for i in (quality_report.errors + quality_report.warnings)[:12])
        parts.append(f'<div class="panel"><ul style="margin:0;padding-left:18px">'
                     f"{items}</ul></div>")
    return "".join(parts)


def _frame_table(frame: pd.DataFrame, numeric: set[str] | None = None) -> str:
    if frame is None or frame.empty:
        return '<div class="empty">no data</div>'
    numeric = numeric or {c for c in frame.columns
                          if pd.api.types.is_numeric_dtype(frame[c])}
    head = "".join(f'<th class="num">{esc(c)}</th>' if c in numeric
                   else f"<th>{esc(c)}</th>" for c in frame.columns)
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row):
            text = (f"{value:.3f}" if isinstance(value, float)
                    else ("" if value is None else str(value)))
            cells.append(f'<td class="num">{esc(text)}</td>' if column in numeric
                         else f"<td>{esc(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def _model_view(backtest: dict | None) -> str:
    """Calibration and the comparison against baselines.

    The most important panel in the dashboard and the least exciting. Every
    price elsewhere is worthless unless the model beats a devigged closing line
    out of sample, and this is where that is visible.
    """
    if not backtest:
        return ('<div class="panel"><div class="empty">'
                "Not run. Add <code>--backtest-from 2019-08-01</code> to include "
                "calibration and the comparison against baselines.<br><br>"
                "Until then nothing here tells you whether the prices in the other "
                "tabs are any good.</div></div>")

    parts = ["<h2>Out-of-sample calibration</h2>"]
    table = backtest.get("calibration")
    if table is not None and not table.empty:
        points = [(float(r.mean_predicted), float(r.observed_rate))
                  for r in table.itertuples()]
        parts.append(
            '<div class="panel">'
            + line_chart(
                [Series("model", points, slot=1)],
                width=640, height=250,
                x_ticks=[(v / 10, f"{v / 10:.1f}") for v in range(0, 11, 2)],
                y_format="{:.2f}",
                reference=[(0.0, 0.0), (1.0, 1.0)],
                reference_label="perfect")
            + '<div class="caption">Predicted probability against how often it '
              "actually happened. Points below the dashed line mean the model is "
              "overconfident there &mdash; and that is exactly where a value bet "
              "would be a losing bet.</div></div>")

    rows = backtest.get("comparison")
    if rows:
        parts.append("<h2>Ranked by RPS</h2>")
        parts.append(
            '<div class="panel">'
            + hbar_chart(rows, width=640, value_format="{:.5f}",
                         highlight="market", lower_is_better=True)
            + '<div class="caption">Ranked probability score, which respects the '
              "ordering of home&ndash;draw&ndash;away: predicting an away win when "
              "the home side won is a worse error than predicting a draw. A model "
              "that cannot beat <b>market</b> has no betting edge, whatever its "
              "expected value says.</div></div>")

    metrics = backtest.get("metrics")
    if metrics:
        tiles = [
            stat_tile(f"{metrics.get('n', 0):,}", "matches scored"),
            stat_tile(f"{metrics.get('rps', float('nan')):.5f}", "RPS",
                      detail="lower is better"),
            stat_tile(f"{metrics.get('ece', float('nan')):.4f}", "calibration error",
                      detail="mean gap, predicted vs actual"),
            stat_tile(backtest.get("verdict", "—"), "vs market",
                      status=backtest.get("verdict_status")),
        ]
        parts.insert(0, f'<div class="tiles">{"".join(tiles)}</div>')
    return "".join(parts)


def _props_view(prop_picks: pd.DataFrame) -> str:
    if prop_picks is None or prop_picks.empty:
        return ('<div class="panel"><div class="empty">'
                "No player data. Run <code>bet ingest --source fbref</code>."
                "</div></div>")

    keep = [c for c in ("player_id", "team_id", "opponent_id", "line",
                        "expected_minutes", "expected", "p_over", "fair_over",
                        "sample_90s") if c in prop_picks.columns]
    frame = prop_picks[keep].copy()
    frame["player_id"] = frame["player_id"].map(lambda p: _short(p).title())
    for column in ("team_id", "opponent_id"):
        if column in frame.columns:
            frame[column] = frame[column].map(_team)
    frame = frame.rename(columns={
        "player_id": "player", "team_id": "team", "opponent_id": "opponent",
        "expected_minutes": "minutes", "expected": "expected",
        "p_over": "P(over)", "fair_over": "fair", "sample_90s": "90s"})

    return ('<div class="panel">' + _frame_table(frame)
            + '<div class="caption">A low <b>90s</b> means the rate is mostly '
              "prior rather than evidence, so the price is a guess dressed as a "
              "number. No historical prop lines are available free, so these "
              "cannot be backtested the way match prices can.</div></div>")


CONTROLS = """
<div class="controls">
  <button class="btn" id="refresh-btn">Refresh data</button>
  <span id="refresh-status">served locally &middot; the page rebuilds on reload</span>
  <button class="btn secondary" id="reload-btn">Reload page</button>
</div>
"""

SERVED_SCRIPT = """
(function(){
  var btn = document.getElementById('refresh-btn');
  var reload = document.getElementById('reload-btn');
  var status = document.getElementById('refresh-status');
  if(!btn) return;

  function say(text, tone){
    status.innerHTML = text;
    status.className = tone === true ? 'err' : (tone || '');
  }

  function detailList(list){
    if(!list || !list.length) return '';
    return '<ul class="errlist">' + list.map(function(e){
      return '<li>' + String(e).replace(/[<&]/g, function(c){
        return c === '<' ? '&lt;' : '&amp;'; }) + '</li>';
    }).join('') + '</ul>';
  }

  // Poll while a fetch runs. The refresh happens on a worker thread server
  // side, so the request returns at once and this reports progress; without
  // it a click would look like nothing happened for a minute.
  function poll(){
    fetch('/api/status').then(function(r){ return r.json(); }).then(function(s){
      if(s.refreshing){
        // Naming the source makes a slow run legible. "fetching from the
        // sources" for ten minutes is indistinguishable from a hang.
        var where = s.progress ? ' &mdash; ' + s.progress : '';
        say('<span class="spin"></span>fetching from the sources' + where);
        setTimeout(poll, 1500);
        return;
      }
      btn.disabled = false;
      if(!s.last_error && s.last_warning){
        // An unreachable source is an outage to note, not a fault to fix.
        // Red here had people looking for a bug in their own install.
        say(s.last_warning + detailList(s.last_errors), 'warn');
        reload.style.display = '';
        return;
      }
      if(s.last_error){
        // Every source failure, not only the first: they are usually
        // independent, and one at a time means one fix per round trip.
        say('refresh failed: ' + s.last_error + detailList(s.last_errors), true);
      } else {
        say('data refreshed &middot; reload to see it');
        reload.style.display = '';
      }
    }).catch(function(){
      btn.disabled = false;
      say('lost contact with the server', true);
    });
  }

  btn.addEventListener('click', function(){
    btn.disabled = true;
    say('<span class="spin"></span>starting...');
    fetch('/api/refresh', {method:'POST'}).then(function(r){ return r.json(); })
      .then(function(d){
        if(!d.started){ btn.disabled = false; say(d.reason || 'could not start', true); return; }
        poll();
      })
      .catch(function(){ btn.disabled = false; say('could not reach the server', true); });
  });

  reload.addEventListener('click', function(){ location.reload(); });
  reload.style.display = 'none';
  poll();
})();
"""


def _previous_matchday_panel(previous, details: dict | None = None) -> str:
    """A collapsed, expand-on-demand look back at the last matchday.

    Closed by default so the page opens on what is coming up, not what
    already happened. `<details>` needs no script and keeps its open/closed
    state across a reload for free, unlike the click-to-expand fixture cards
    which reset on every render.
    """
    if previous is None or not previous.matches:
        return ""

    details = details or {}
    scored = [m for m in previous.matches if m.predicted_correct is not None]
    record = (f" &middot; {sum(1 for m in scored if m.predicted_correct)}/"
             f"{len(scored)} correct" if scored else "")
    cards = "".join(_history_card(m, details.get(m.match_id), i)
                    for i, m in enumerate(previous.matches))

    return (
        '<details class="history">'
        f'<summary>Previous matchday{record}</summary>'
        f'<div class="historybody">{cards}</div>'
        "</details>")


def render(store, brief, *, previous=None, previous_details: dict | None = None,
           coverage=None, quality_report=None,
           backtest: dict | None = None, details: dict | None = None,
           provenance: dict | None = None, as_of: datetime | None = None,
           served: bool = False) -> str:
    """Assemble the page.

    `served` adds the refresh control. It is off for a written file, where the
    button would post to a server that is not there.
    """
    generated = datetime.utcnow()
    as_of = as_of or generated
    details = details or {}
    banner = _provenance_banner(provenance, as_of) if provenance else ""
    controls = CONTROLS if served else ""
    served_script = SERVED_SCRIPT if served else ""

    history_panel = _previous_matchday_panel(previous, previous_details)
    cards = "".join(_fixture_card(m, details.get(m.match_id), i)
                    for i, m in enumerate(brief.matches))
    fixtures_view = history_panel + (cards or ('<div class="panel"><div class="empty">'
                                     "No fixtures in the window.</div></div>"))

    views = [
        ("overview", "Overview", _overview(brief, coverage, quality_report)),
        ("fixtures", f"Fixtures ({len(brief.matches)})", fixtures_view),
        ("model", "Model health", _model_view(backtest)),
        ("props", "Props", _props_view(brief.prop_picks)),
    ]

    tabs = "".join(
        f'<button data-view="{vid}" aria-selected="{"true" if i == 0 else "false"}">'
        f"{esc(title)}</button>" for i, (vid, title, _) in enumerate(views))
    panes = "".join(
        f'<section class="view{" on" if i == 0 else ""}" id="{vid}">{body}</section>'
        for i, (vid, _, body) in enumerate(views))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bundesliga model</title><style>{STYLE}</style></head>
<body><div id="tip"></div>
<div id="player-modal-backdrop" hidden>
  <div id="player-modal" role="dialog" aria-modal="true" aria-labelledby="pm-name">
    <button id="pm-close" aria-label="Close">&times;</button>
    <h3 id="pm-name"></h3>
    <div id="pm-team" class="kick"></div>
    <div id="pm-rows" class="meta" style="display:block;margin-top:10px"></div>
    <div id="pm-note" class="note"></div>
  </div>
</div>
<div class="wrap" style="position:relative">
<header><h1>Bundesliga model</h1>
<div class="sub">Generated {generated:%Y-%m-%d %H:%M} UTC &middot;
{len(brief.matches)} fixture(s) {brief.scope_label} &middot;
click a fixture for line-ups and squad stats</div>
{controls}{banner}
<nav role="tablist">{tabs}</nav></header>
{panes}
<footer>Fair odds and expected value are net of a {GERMAN_STAKE_TAX:.1%}
betting-stake tax and bookmaker margin. Probabilities come from a time-weighted
Dixon-Coles fit priced off the expected eleven. A scoreline is shown only when it
is clearly the most likely one &mdash; a blank means the model cannot separate
the leading scores, which is most matches. A model edge is not a proven edge:
check the model-health tab before acting on anything here.</footer>
</div><script>{SCRIPT}{served_script}</script></body></html>"""


def build(store, as_of: datetime | None = None, *, days: int = 8,
          league: str = "bundesliga", include_quality: bool = True,
          backtest_from: str | None = None, served: bool = False) -> str:
    """Compile a brief, gather per-match detail, and render.

    The fixtures shown are the next matchday, not a fixed lookahead: the
    Bundesliga does not play every week, and a fixed window (the previous
    behaviour of `days` here) regularly landed empty between matchdays --
    reported as "no matches displayed" when the store was in fact fully
    loaded. Matchday scoping cannot go blank that way as long as a next
    matchday exists at all. `days` still bounds how far back the previous
    matchday is searched for, which only matters across an unusually long
    gap (the summer break).

    Above the upcoming fixtures, a collapsed panel holds the previous
    matchday: what the model would genuinely have said beforehand (fit
    strictly before it kicked off, never on data that includes its own
    result) next to what actually happened.
    """
    from bet.recommend import next_matchday_brief, previous_matchday_brief

    as_of = as_of or datetime.utcnow()
    brief = next_matchday_brief(store, as_of, league=league)
    previous = previous_matchday_brief(
        store, as_of, league=league, search_days=max(days, 60))

    coverage, quality_report = None, None
    if include_quality:
        from bet.quality import run_quality_checks
        quality_report = run_quality_checks(store, as_of)
        coverage = quality_report.coverage

    names = store.con.execute("SELECT player_id, full_name FROM player").df()
    name_map = dict(zip(names["player_id"], names["full_name"])) if not names.empty else {}

    details = _build_match_details(store, brief.matches, as_of, name_map)
    # Built at "now", unlike the predictions in the cards themselves. The
    # detail panel is the historical record, not a graded forecast: the
    # probabilities and expected goals stay exactly what the model would
    # genuinely have said before kickoff (fit at `previous.as_of`), but the
    # confirmed line-up that actually played is worth showing as what it is
    # once it is known, rather than freezing the lookup at the same instant
    # for no reason -- the same way the actual score is shown next to the
    # prediction rather than withheld.
    previous_details = _build_match_details(store, previous.matches, as_of, name_map)

    backtest = _run_backtest(store, backtest_from, league) if backtest_from else None
    return render(store, brief, previous=previous, previous_details=previous_details,
                  coverage=coverage, quality_report=quality_report, backtest=backtest,
                  details=details, provenance=data_provenance(store, as_of), as_of=as_of,
                  served=served)


def _build_match_details(store, matches: list, as_of: datetime,
                         name_map: dict[str, str]) -> dict:
    """Formation, lineup and scoreline detail for a list of recommendations,
    all read as of the same moment used to price them."""
    if not matches:
        return {}

    from bet.models.dixon_coles import DixonColesModel
    model = DixonColesModel(use_availability=True).fit(store, as_of)
    if model.params is None:
        return {}

    rates = store.player_rates_as_of(as_of, min_minutes=0.0)
    return {match.match_id: build_match_detail(store, model, match, as_of, rates, name_map)
            for match in matches}


def _run_backtest(store, start: str, league: str) -> dict | None:
    """Walk models forward so the model-health tab has something in it."""
    from bet.evaluation.backtest import compare
    from bet.models.baselines import EloModel, HomePriorModel, MarketModel
    from bet.models.dixon_coles import DixonColesModel

    table, results = compare(
        store,
        [HomePriorModel(), EloModel(), DixonColesModel(), MarketModel()],
        start=datetime.fromisoformat(start), league=league)
    if table.empty:
        return None

    best = results.get("dixon_coles_goals")
    market_rps = float(table.loc[table["model"] == "market", "rps"].iloc[0]) \
        if "market" in table["model"].values else None
    model_rps = float(table.loc[table["model"] == "dixon_coles_goals", "rps"].iloc[0]) \
        if "dixon_coles_goals" in table["model"].values else None

    verdict, status = "—", None
    if market_rps is not None and model_rps is not None:
        if model_rps < market_rps:
            verdict, status = "beats market", "good"
        else:
            gap = (model_rps - market_rps) / market_rps
            verdict, status = f"{gap:.1%} worse", "warning"

    return {
        "comparison": [(str(r.model), float(r.rps)) for r in table.itertuples()],
        "calibration": best.calibration if best is not None else None,
        "metrics": best.metrics if best is not None else None,
        "verdict": verdict,
        "verdict_status": status,
    }


# ---------------------------------------------------------------- provenance


# Sources that mean "this is not real data". A dashboard built from fixtures
# looks exactly like one built from Bundesliga results, which is how a demo gets
# mistaken for a forecast.
SYNTHETIC_SOURCES = {"synthetic", "demo", "test", "fixture"}

# How old the newest row may be before the numbers stop being "current". A week
# covers a normal gap between matchdays; beyond that an ingest has probably
# stopped working and nobody has noticed.
STALE_AFTER_DAYS = 8


def data_provenance(store, as_of: datetime) -> dict:
    """Where the numbers came from and how old they are.

    Reported on the page rather than kept in a log, because the failure this
    prevents is silent: a scraper stops working, the model keeps running on an
    ageing snapshot, and every price stays plausible while quietly going stale.
    """
    sources = store.con.execute(
        "SELECT DISTINCT source FROM match_result WHERE known_at <= ?",
        [as_of]).df()
    names = set(sources["source"].dropna()) if not sources.empty else set()
    synthetic = bool(names) and names <= SYNTHETIC_SOURCES

    freshness = []
    for table, label in (("match_result", "results"),
                         ("player_match_stat", "player stats"),
                         ("odds_quote", "odds"),
                         ("lineup", "line-ups"),
                         ("player_availability", "team news")):
        try:
            # Filtered to what was knowable at as_of, like every other read in
            # this project. Without it a store holding future fixtures reports
            # a negative age, which is both wrong and obviously wrong.
            newest = store.con.execute(
                f"SELECT MAX(known_at) FROM {table} WHERE known_at <= ?",
                [as_of]).fetchone()[0]
        except Exception:
            newest = None
        if newest is None:
            freshness.append({"label": label, "newest": None, "age_days": None})
            continue
        age = (pd.Timestamp(as_of) - pd.Timestamp(newest)).days
        freshness.append({"label": label, "newest": pd.Timestamp(newest),
                          "age_days": age})

    populated = [f for f in freshness if f["age_days"] is not None]
    worst = max((f["age_days"] for f in populated), default=None)

    # How much the model actually has to learn from, which is not the same
    # question as how fresh the newest row is: a store refreshed ten minutes
    # ago can still hold one matchday and price nothing.
    from bet.live import MIN_MATCHES_FOR_A_MODEL
    try:
        played = int(store.con.execute(
            "SELECT count(*) FROM match m JOIN match_result r USING (match_id) "
            "WHERE r.known_at <= ?", [as_of]).fetchone()[0])
    except Exception:
        played = 0

    return {
        "sources": sorted(names),
        "synthetic": synthetic,
        "freshness": freshness,
        "worst_age_days": worst,
        "stale": worst is not None and worst > STALE_AFTER_DAYS,
        "empty": not populated,
        "played_matches": played,
        "thin": bool(populated) and played < MIN_MATCHES_FOR_A_MODEL,
    }


def _provenance_banner(provenance: dict, as_of: datetime) -> str:
    """A visible statement of what this page is built from."""
    if provenance["empty"]:
        return ('<div class="provenance stale"><b>&#9888; No data</b>'
                "The store is empty. Press <b>Refresh data</b>, or run "
                "<code>bet ingest --source football_data --seasons 2015-2026</code>"
                ".</div>")

    if provenance.get("thin"):
        # Fixtures without expectations, and nothing saying why, is the worst
        # of both: the page looks broken when it is merely un-backfilled.
        played = provenance.get("played_matches", 0)
        return ('<div class="provenance stale"><b>&#9888; Not enough history '
                "to price anything</b>"
                f"The store holds {played} played match(es). The model fits an "
                "attack and a defence for every club, which needs seasons, not "
                "matchdays &mdash; so fixtures are listed below but every "
                "expectation is blank. Press <b>Refresh data</b> to fetch the "
                "back catalogue (about a minute), or run "
                "<code>bet ingest --source football_data --seasons 2015-2026"
                "</code>.</div>")

    rows = []
    for entry in provenance["freshness"]:
        if entry["age_days"] is None:
            rows.append(f'<span>{esc(entry["label"])}: <b>none</b></span>')
            continue
        age = entry["age_days"]
        cls = ' class="stale-cell"' if age > STALE_AFTER_DAYS else ""
        rows.append(f'<span{cls}>{esc(entry["label"])}: '
                    f'{entry["newest"]:%d %b %Y} ({age}d)</span>')
    detail = f'<div class="meta" style="margin-top:6px">{"".join(rows)}</div>'

    if provenance["synthetic"]:
        # The loudest thing on the page, deliberately. Names, stats and
        # scorelines below are invented and must not be read as a forecast.
        return (
            '<div class="provenance synthetic">'
            "<b>&#10007; Synthetic data &mdash; not a real forecast</b>"
            f"Every source in this store is a test fixture "
            f"({esc(', '.join(provenance['sources']))}). The players, stats and "
            "prices below are generated, not Bundesliga data. Run "
            "<code>bet ingest --source all</code> against the real sources before "
            "using any of this."
            f"{detail}</div>")

    if provenance["stale"]:
        return (
            '<div class="provenance stale">'
            f"<b>&#9888; Data is {provenance['worst_age_days']} days old</b>"
            "Some of the numbers below predate the most recent matchday, so the "
            "model is running on an ageing snapshot. Re-run "
            "<code>bet ingest</code>."
            f"{detail}</div>")

    return (f'<div class="provenance"><b>Data current as of '
            f"{as_of:%d %b %Y}</b>"
            f"Sources: {esc(', '.join(provenance['sources']))}."
            f"{detail}</div>")
