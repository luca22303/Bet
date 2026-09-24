"""Inline SVG chart primitives.

No charting library. The dashboard is a single file that must open from disk
with no network, so every mark is hand-built SVG. That constraint is cheap here
because the chart vocabulary is small: a diverging bar, a line, a horizontal
bar, a scatter.

Colours come from the validated reference palette and are referenced through CSS
custom properties rather than literal hex, so light and dark swap in one place.
The palette was checked with the validator rather than by eye: the categorical
trio passes all-pairs in both modes, and the home/away poles clear CVD
separation comfortably (delta-E 21.6 light, 19.2 dark).

Two rules shape everything below. Text never wears a series colour -- values and
labels stay in ink tokens with a coloured mark beside them carrying identity.
And marks that touch get a 2px surface-coloured gap, so adjacent segments read
as separate rather than merging into one block.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}"


# --------------------------------------------------------------- diverging bar


def diverging_bar(probabilities: dict[str, float], labels: dict[str, str],
                  *, height: int = 26, min_label_share: float = 0.09) -> str:
    """A 1X2 probability bar.

    Home and away are opposite outcomes, so they take the diverging poles (blue
    and red) with the draw on the neutral grey midpoint. Three categorical hues
    would imply the outcomes are unordered identities, which they are not --
    reading left-to-right is reading home-win through draw to away-win.

    Segments are separated by a 2px surface gap so they do not merge, and a
    share too small to hold a legible label is left bare rather than clipped.
    """
    order = (("H", "seg-home"), ("D", "seg-draw"), ("A", "seg-away"))
    cells = []
    for outcome, css in order:
        share = max(0.0, float(probabilities.get(outcome, 0.0)))
        label = f"{share:.0%}" if share >= min_label_share else ""
        cells.append(
            f'<div class="seg {css}" style="flex:{share * 1000:.3f} 0 0"'
            f' data-tip="{esc(labels.get(outcome, outcome))} &middot; {share:.1%}">'
            f"<span>{label}</span></div>")
    return f'<div class="dbar" style="height:{height}px">{"".join(cells)}</div>'


# ---------------------------------------------------------------- line chart


@dataclass
class Series:
    name: str
    points: list[tuple[float, float]]
    slot: int = 1          # categorical slot, 1-indexed
    dashed: bool = False


def line_chart(series: list[Series], *, width: int = 640, height: int = 220,
               x_label: str = "", y_label: str = "", x_ticks: list[tuple[float, str]] | None = None,
               y_format: str = "{:.2f}", reference: list[tuple[float, float]] | None = None,
               reference_label: str = "") -> str:
    """A multi-series line chart on one axis.

    One axis, always. Two measures on different scales get two charts or a
    common index -- a second y-scale invents a correlation the data does not
    contain, which is the single most common way a chart misleads.
    """
    usable = [s for s in series if len(s.points) >= 2]
    if not usable:
        return '<div class="empty">not enough data to plot</div>'

    pad_l, pad_r, pad_t, pad_b = 46, 14, 12, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    xs = [p[0] for s in usable for p in s.points]
    ys = [p[1] for s in usable for p in s.points]
    if reference:
        xs += [p[0] for p in reference]
        ys += [p[1] for p in reference]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    # A little headroom stops the top mark colliding with the frame.
    span = (y_max - y_min) or 1.0
    y_min, y_max = y_min - span * 0.08, y_max + span * 0.08

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return pad_t + (1 - (y - y_min) / (y_max - y_min)) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']

    # Gridlines and y ticks, recessive.
    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        gy = sy(y)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{gy:.1f}" '
                     f'x2="{width - pad_r}" y2="{gy:.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 7}" y="{gy + 3.5:.1f}" '
                     f'text-anchor="end">{esc(y_format.format(y))}</text>')

    if x_ticks:
        for x, label in x_ticks:
            parts.append(f'<text class="tick" x="{sx(x):.1f}" y="{height - 9}" '
                         f'text-anchor="middle">{esc(label)}</text>')

    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" '
                 f'x2="{pad_l}" y2="{pad_t + plot_h}"/>')

    if reference and len(reference) >= 2:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                        for i, (x, y) in enumerate(reference))
        parts.append(f'<path class="reference" d="{path}"/>')
        rx, ry = reference[-1]
        parts.append(f'<text class="reflabel" x="{sx(rx) - 6:.1f}" y="{sy(ry) - 7:.1f}" '
                     f'text-anchor="end">{esc(reference_label)}</text>')

    for s in usable:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                        for i, (x, y) in enumerate(s.points))
        dash = ' stroke-dasharray="5 4"' if s.dashed else ""
        parts.append(f'<path class="line s{s.slot}" d="{path}"{dash}/>')
        # Markers are >=8px so they are a real hover target, and carry a 2px
        # surface ring so overlapping points stay countable.
        for x, y in s.points:
            parts.append(
                f'<circle class="dot s{s.slot}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4"'
                f' data-tip="{esc(s.name)} &middot; {esc(y_format.format(y))}"/>')
        # Direct label at the series end: identity never rests on colour alone.
        lx, ly = s.points[-1]
        parts.append(f'<text class="serieslabel s{s.slot}-ink" x="{sx(lx) - 4:.1f}" '
                     f'y="{sy(ly) - 9:.1f}" text-anchor="end">{esc(s.name)}</text>')

    if x_label:
        parts.append(f'<text class="axislabel" x="{pad_l + plot_w / 2:.0f}" '
                     f'y="{height - 1}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------- horizontal bars


def hbar_chart(rows: list[tuple[str, float]], *, width: int = 640,
               bar_height: int = 22, gap: int = 8, value_format: str = "{:.4f}",
               highlight: str | None = None, lower_is_better: bool = True) -> str:
    """Ranked horizontal bars, one colour.

    One series means one hue. Shading each bar by its own value would burn the
    only free channel restating the length, and would fail the categorical
    checks by construction.

    `highlight` marks the benchmark row -- the thing everything else is measured
    against -- using weight and a label rather than a second hue.
    """
    if not rows:
        return '<div class="empty">no results</div>'

    label_w, value_w = 150, 74
    plot_w = width - label_w - value_w
    height = len(rows) * (bar_height + gap) + gap
    largest = max(abs(v) for _, v in rows) or 1.0

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for i, (name, value) in enumerate(rows):
        y = gap + i * (bar_height + gap)
        bar_w = max(2.0, abs(value) / largest * plot_w)
        is_mark = highlight is not None and name == highlight
        cls = "bar benchmark" if is_mark else "bar"
        parts.append(
            f'<text class="rowlabel{" strong" if is_mark else ""}" x="{label_w - 10}" '
            f'y="{y + bar_height * 0.72:.0f}" text-anchor="end">{esc(name)}</text>')
        # 4px rounded data-end, anchored square to the baseline.
        parts.append(
            f'<rect class="{cls}" x="{label_w}" y="{y}" width="{bar_w:.1f}" '
            f'height="{bar_height}" rx="4"'
            f' data-tip="{esc(name)} &middot; {esc(value_format.format(value))}"/>')
        parts.append(
            f'<text class="barvalue" x="{label_w + bar_w + 8:.1f}" '
            f'y="{y + bar_height * 0.72:.0f}">{esc(value_format.format(value))}</text>')
        if is_mark:
            parts.append(
                f'<text class="benchlabel" x="{label_w + bar_w + 8 + 58:.1f}" '
                f'y="{y + bar_height * 0.72:.0f}">benchmark</text>')
    parts.append("</svg>")

    direction = "lower is better" if lower_is_better else "higher is better"
    return "".join(parts) + f'<div class="caption">{direction}</div>'


# -------------------------------------------------------------------- scatter


def scatter(points: list[dict], *, width: int = 640, height: int = 300,
            x_label: str = "", y_label: str = "", invert_y: bool = False,
            quadrant_labels: tuple[str, str, str, str] | None = None) -> str:
    """Labelled scatter for two measures.

    Each point carries its own label, because a legend mapping eighteen clubs to
    eighteen hues would need eighteen distinguishable colours -- past eight there
    are none, and the rule is to fold or facet rather than generate more.
    """
    if not points:
        return '<div class="empty">no data</div>'

    pad_l, pad_r, pad_t, pad_b = 50, 18, 16, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = (x_max - x_min) * 0.12 or 0.1
    y_pad = (y_max - y_min) * 0.12 or 0.1
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        t = (y - y_min) / (y_max - y_min)
        return pad_t + (t if invert_y else 1 - t) * plot_h

    x_mid = sum(xs) / len(xs)
    y_mid = sum(ys) / len(ys)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    # Mean crosshairs give the quadrants meaning without a second colour scale.
    parts.append(f'<line class="grid dashed" x1="{sx(x_mid):.1f}" y1="{pad_t}" '
                 f'x2="{sx(x_mid):.1f}" y2="{pad_t + plot_h}"/>')
    parts.append(f'<line class="grid dashed" x1="{pad_l}" y1="{sy(y_mid):.1f}" '
                 f'x2="{width - pad_r}" y2="{sy(y_mid):.1f}"/>')

    if quadrant_labels:
        tl, tr, bl, br = quadrant_labels
        parts.append(f'<text class="quad" x="{pad_l + 6}" y="{pad_t + 13}">{esc(tl)}</text>')
        parts.append(f'<text class="quad" x="{width - pad_r - 6}" y="{pad_t + 13}" '
                     f'text-anchor="end">{esc(tr)}</text>')
        parts.append(f'<text class="quad" x="{pad_l + 6}" y="{pad_t + plot_h - 5}">{esc(bl)}</text>')
        parts.append(f'<text class="quad" x="{width - pad_r - 6}" y="{pad_t + plot_h - 5}" '
                     f'text-anchor="end">{esc(br)}</text>')

    for point in points:
        cx, cy = sx(point["x"]), sy(point["y"])
        parts.append(
            f'<circle class="pt" cx="{cx:.1f}" cy="{cy:.1f}" r="5"'
            f' data-tip="{esc(point.get("tip", point.get("label", "")))}"/>')
        parts.append(f'<text class="ptlabel" x="{cx:.1f}" y="{cy - 9:.1f}" '
                     f'text-anchor="middle">{esc(point.get("label", ""))}</text>')

    parts.append(f'<text class="axislabel" x="{pad_l + plot_w / 2:.0f}" y="{height - 4}" '
                 f'text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text class="axislabel" x="14" y="{pad_t + plot_h / 2:.0f}" '
                 f'text-anchor="middle" transform="rotate(-90 14 {pad_t + plot_h / 2:.0f})">'
                 f"{esc(y_label)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def stat_tile(value: str, label: str, *, detail: str = "", status: str | None = None) -> str:
    """A single number.

    The most common charting mistake is drawing eight hues when the story is one
    number. A one-bar bar chart is the same mistake with fewer colours -- when
    the number *is* the chart, this is the right form.

    `status` carries an icon and a word alongside the colour, never the colour
    alone.
    """
    icons = {"good": "&#10003;", "warning": "&#9888;", "critical": "&#10007;"}
    badge = ""
    if status:
        badge = (f'<span class="badge {status}">{icons.get(status, "")} '
                 f'{esc(status)}</span>')
    detail_html = f'<div class="tiledetail">{esc(detail)}</div>' if detail else ""
    return (f'<div class="tile"><div class="tilevalue">{esc(value)}</div>'
            f'<div class="tilelabel">{esc(label)} {badge}</div>{detail_html}</div>')


# ---------------------------------------------------------------------- pitch


def parse_formation_rows(formation: str) -> list[int]:
    """'4-2-3-1' -> [4, 2, 3, 1], the outfield lines back to front.

    The display uses the literal lines rather than the defender/midfield/forward
    totals, because 4-2-3-1 and 4-3-3 field the same five midfielders in
    visibly different shapes, and the shape is the point of drawing a pitch.
    """
    try:
        rows = [int(part) for part in formation.split("-") if part.strip()]
    except ValueError:
        return [4, 2, 3, 1]
    return rows if rows and sum(rows) == 10 else [4, 2, 3, 1]


def assign_to_formation(formation: str, players: list[dict]) -> list[dict]:
    """Place players on the lines of a formation.

    Defenders fill the back line, forwards the front, midfielders the rest. A
    squad that does not fit the named shape still gets placed rather than
    dropped -- an incomplete pitch is more useful than an empty one.

    Every player is placed at most once. The obvious implementation keeps a
    per-group pool plus a shared spare pool, but those hold the same objects, so
    taking a midfielder from his group leaves him in the spares and he appears
    twice on the pitch. Selection here goes through one `used` set, which makes
    that impossible rather than merely unlikely.
    """
    rows = parse_formation_rows(formation)
    used: set[int] = set()

    def take(preferred: str | None) -> dict | None:
        """The next unused player, preferring a position group."""
        if preferred is not None:
            for player in players:
                if id(player) not in used and player.get("group") == preferred:
                    used.add(id(player))
                    return player
        for player in players:
            if id(player) not in used and player.get("group") != "goalkeeper":
                used.add(id(player))
                return player
        return None

    placed: list[dict] = []
    keeper = take("goalkeeper")
    if keeper is not None:
        placed.append({**keeper, "row": -1, "col": 0, "row_size": 1})

    for row_index, count in enumerate(rows):
        if row_index == 0:
            group = "defender"
        elif row_index == len(rows) - 1:
            group = "forward"
        else:
            group = "midfielder"
        for col in range(count):
            player = take(group)
            if player is None:
                continue
            placed.append({**player, "row": row_index, "col": col, "row_size": count})
    return placed


# The validated sequential pair from the reference palette (--seq-100 /
# --seq-450 in the dashboard's own CSS), interpolated here in Python rather
# than with CSS color-mix so the label contrast below can be computed against
# the exact colour actually drawn, band by band.
_SEQ_LOW = (0xCD, 0xE2, 0xFB)
_SEQ_HIGH = (0x2A, 0x78, 0xD6)


def _seq_color(intensity: float) -> tuple[str, str]:
    """A colour along the sequential scale for `intensity` in [0, 1], and
    which ink -- dark or the surface white -- reads on top of it."""
    t = max(0.0, min(1.0, intensity))
    rgb = tuple(round(lo + (hi - lo) * t) for lo, hi in zip(_SEQ_LOW, _SEQ_HIGH))

    def channel(v: int) -> float:
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * channel(rgb[0]) + 0.7152 * channel(rgb[1]) + 0.0722 * channel(rgb[2])
    ink = "var(--ink)" if luminance > 0.4 else "#fff"
    return "#{:02x}{:02x}{:02x}".format(*rgb), ink


def zone_heatmap(shares: dict, *, width: int = 300, height: int = 400,
                 team_name: str = "") -> str:
    """Three horizontal bands, shaded by share of touches, attacking upward.

    `shares` is `bet.spatial.pitch.touch_zone_shares`'s own return shape.
    Coarse by construction -- FBref's zone breakdown has three bands, not a
    continuous surface -- so this reads as a small number of honestly blocky
    regions rather than a smoothed density the data cannot support. Penalty-
    area touches are shown as a count beside each box instead of a fourth and
    fifth band: they are subsets of the thirds already drawn, not disjoint
    from them.
    """
    thirds = shares.get("thirds", {})
    penalty = shares.get("penalty", {})

    margin = 16
    play_w = width - margin * 2
    play_h = height - margin * 2
    band_h = play_h / 3.0

    parts = [f'<svg viewBox="0 0 {width} {height}" class="pitch" role="img" '
             f'aria-label="{esc(team_name)} touch zones">']
    parts.append(f'<rect class="turf" x="{margin}" y="{margin}" width="{play_w}" '
                 f'height="{play_h}" rx="4"/>')

    # Attacking third at the top, matching the formation pitch's own
    # "attacking upward" convention so the two read as the same orientation.
    for i, (key, label) in enumerate((("att", "Attacking third"),
                                      ("mid", "Middle third"),
                                      ("def", "Defensive third"))):
        share = thirds.get(key, 0.0)
        y = margin + i * band_h
        # A third with an exactly even share of the ball is ~33%; 55% is
        # already a heavy concentration, so the scale saturates there rather
        # than needing an implausible 100% to reach full colour.
        fill, ink = _seq_color(share / 0.55)
        parts.append(f'<rect x="{margin}" y="{y:.1f}" width="{play_w}" height="{band_h:.1f}" '
                     f'fill="{fill}" data-tip="{esc(label)} &middot; {share:.0%} of touches"/>')
        parts.append(f'<text x="{width / 2}" y="{y + band_h / 2 + 4:.1f}" text-anchor="middle" '
                     f'font-size="13" font-weight="600" fill="{ink}">{share:.0%}</text>')

    # Pitch markings drawn last, over the bands, so they still read.
    mid_y = margin + play_h / 2
    parts.append(f'<line class="pitchline" x1="{margin}" y1="{mid_y:.1f}" '
                 f'x2="{margin + play_w}" y2="{mid_y:.1f}"/>')
    parts.append(f'<circle class="pitchline" cx="{width / 2}" cy="{mid_y:.1f}" r="26" fill="none"/>')
    box_w, box_h = play_w * 0.56, play_h * 0.12
    box_x = (width - box_w) / 2
    parts.append(f'<rect class="pitchline" x="{box_x:.1f}" y="{margin:.1f}" '
                 f'width="{box_w:.1f}" height="{box_h:.1f}" fill="none"/>')
    parts.append(f'<rect class="pitchline" x="{box_x:.1f}" y="{margin + play_h - box_h:.1f}" '
                 f'width="{box_w:.1f}" height="{box_h:.1f}" fill="none"/>')

    if "att_pen" in penalty:
        parts.append(f'<text class="benchlabel" x="{width / 2}" y="{margin + box_h + 12:.1f}" '
                     f'text-anchor="middle">{int(penalty["att_pen"])} in the box</text>')
    if "def_pen" in penalty:
        parts.append(f'<text class="benchlabel" x="{width / 2}" '
                     f'y="{margin + play_h - box_h - 6:.1f}" text-anchor="middle">'
                     f'{int(penalty["def_pen"])} in the box</text>')

    parts.append("</svg>")
    return "".join(parts)


def pitch(formation: str, players: list[dict], *, width: int = 300,
          height: int = 400, team_name: str = "", confirmed: bool = False) -> str:
    """A formation drawn on a pitch, attacking upward.

    Positions are indicative, not tracked: this shows the shape a manager names,
    not where anyone stood. Drawing it any more precisely than the data supports
    would imply a measurement that does not exist.
    """
    rows = parse_formation_rows(formation)
    placed = assign_to_formation(formation, players)

    margin = 16
    play_w = width - margin * 2
    play_h = height - margin * 2
    lines = len(rows)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="pitch" role="img" '
             f'aria-label="{esc(team_name)} {esc(formation)}">']
    parts.append(f'<rect class="turf" x="{margin}" y="{margin}" width="{play_w}" '
                 f'height="{play_h}" rx="4"/>')
    # Halfway line and centre circle, drawn recessive: they orient the eye and
    # must not compete with the players.
    parts.append(f'<line class="pitchline" x1="{margin}" y1="{margin}" '
                 f'x2="{margin + play_w}" y2="{margin}"/>')
    parts.append(f'<circle class="pitchline" cx="{width / 2}" cy="{margin}" r="30" fill="none"/>')
    box_w, box_h = play_w * 0.56, play_h * 0.16
    parts.append(f'<rect class="pitchline" x="{(width - box_w) / 2:.1f}" '
                 f'y="{margin + play_h - box_h:.1f}" width="{box_w:.1f}" '
                 f'height="{box_h:.1f}" fill="none"/>')

    def position(player: dict) -> tuple[float, float]:
        row, col, size = player["row"], player["col"], max(player["row_size"], 1)
        if row < 0:                                   # goalkeeper
            return width / 2, margin + play_h - 16
        # Lines run from just in front of the keeper to the attacking third.
        depth = (row + 1) / (lines + 1)
        y = margin + play_h - 34 - depth * (play_h - 70)
        x = margin + play_w * (col + 1) / (size + 1)
        return x, y

    for player in placed:
        x, y = position(player)
        propensity = player.get("propensity")
        # A predicted XI is a guess with a confidence, and the mark says so: a
        # dashed ring means the model is unsure this player starts.
        uncertain = (not confirmed and propensity is not None and propensity < 0.55)
        cls = "player uncertain" if uncertain else "player"
        tip = player.get("tip") or player.get("name", "")

        modal = player.get("modal")
        group_open = "<g>"
        group_close = "</g>"
        if modal:
            # The whole mark -- circle, initials, name -- opens the same
            # modal, so the click target is not just the small circle.
            payload = esc(json.dumps(modal))
            group_open = (f'<g class="playermark" role="button" tabindex="0" '
                         f'aria-label="{esc(modal.get("name", ""))} stats" '
                         f'data-player=\'{payload}\'>')

        parts.append(group_open)
        parts.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="11" '
                     f'data-tip="{esc(tip)}"/>')
        parts.append(f'<text class="pnum" x="{x:.1f}" y="{y + 3.5:.1f}" '
                     f'text-anchor="middle">{esc(player.get("short", "")[:3])}</text>')
        parts.append(f'<text class="pname" x="{x:.1f}" y="{y + 23:.1f}" '
                     f'text-anchor="middle">{esc(player.get("name", "")[:12])}</text>')
        parts.append(group_close)

    parts.append("</svg>")
    return "".join(parts)
