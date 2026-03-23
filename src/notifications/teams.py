from __future__ import annotations

import csv
import html as html_mod
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from src.models.versioning import MODEL_VERSION

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


# ── Helpers ────────────────────────────────────────────────────────


def _get_model_modified_at() -> str:
    """Timestamp of the newest model artifact file."""
    try:
        latest = max(
            (
                f.stat().st_mtime
                for f in _ARTIFACTS_DIR.glob("model_*.json")
                if f.exists()
            ),
            default=0,
        )
        if latest > 0:
            return datetime.fromtimestamp(latest, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return "unknown"


def _app_build_stamp() -> str:
    """Container / app build timestamp (set via env or fallback)."""
    return os.environ.get(
        "APP_BUILD_TIMESTAMP",
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


# Minimum edge (in points) for a pick to qualify
MIN_EDGE = 2.0
# League-average total used as baseline when no market line exists
_NBA_AVG_TOTAL = 222.0


def _fire_emojis(edge: float) -> str:
    """Return fire emojis proportional to edge strength."""
    if edge >= 9:
        return " 🔥🔥🔥🔥🔥"
    if edge >= 7:
        return " 🔥🔥🔥🔥"
    if edge >= 5:
        return " 🔥🔥🔥"
    if edge >= 3.5:
        return " 🔥🔥"
    return " 🔥"


def _edge_color(edge: float) -> str:
    """Adaptive Card colour for an edge value."""
    if edge >= 7:
        return "Attention"
    if edge >= 4:
        return "Good"
    return "Accent"


_CST = ZoneInfo("US/Central")


def _fmt_time_cst(dt: datetime | None) -> str:
    """Format a datetime as '6:30 PM CT'."""
    if not dt:
        return "TBD"
    ct = dt.astimezone(_CST)
    raw = ct.strftime("%I:%M %p")
    return raw.lstrip("0") + " CT"


def _team_record(team: Any) -> str:
    """Extract W-L record string from a team object."""
    stats = getattr(team, "season_stats", None)
    if stats:
        s = stats[0] if isinstance(stats, list) and stats else stats
        w = getattr(s, "wins", None)
        loss = getattr(s, "losses", None)
        if w is not None and loss is not None:
            return f"{w}-{loss}"
    w = getattr(team, "wins", None)
    loss = getattr(team, "losses", None)
    if w is not None and loss is not None:
        return f"{w}-{loss}"
    return ""


# ── Pick extraction ────────────────────────────────────────────────


@dataclass(frozen=True)
class Pick:
    """A single actionable pick derived from a prediction."""

    label: str  # e.g. "UNDER 222.5" or "Celtics -3.5"
    edge: float  # positive value, higher = stronger
    time_cst: str  # "6:30 PM CT"
    matchup: str  # "Heat @ Celtics"
    segment: str  # "FG" or "1H"
    market_type: str  # "SPREAD", "TOTAL", "ML"
    market_line: str  # "-3.5", "222.5", "65%"
    model_scores: str  # "112-108"
    home_record: str  # "42-18" or ""
    away_record: str  # "28-32" or ""
    confidence: int  # 1-5 fire count
    odds: str = ""  # American odds, e.g. "-110", "+150"


def _fire_count(edge: float) -> int:
    """Return 1-5 fire level based on edge strength."""
    if edge >= 9:
        return 5
    if edge >= 7:
        return 4
    if edge >= 5:
        return 3
    if edge >= 3.5:
        return 2
    return 1


def extract_picks(
    pred: Any,
    game: Any,
    min_edge: float = MIN_EDGE,
    odds_map: dict[str, str] | None = None,
) -> list[Pick]:
    """Extract actionable picks with edge values from a prediction.

    Each pick includes segment, market line, model scores, team records,
    confidence rating, and odds/price.

    ``odds_map`` is an optional dict keyed by
    ``"<segment>_<MARKET>"`` (e.g. ``"FG_SPREAD"``, ``"1H_TOTAL"``) whose
    values are American-odds strings like ``"-110"`` or ``"+150"``.
    """
    if odds_map is None:
        odds_map = {}
    home_team = game.home_team
    away_team = game.away_team
    home = home_team.name if home_team else f"Team {game.home_team_id}"
    away = away_team.name if away_team else f"Team {game.away_team_id}"
    t_cst = _fmt_time_cst(game.commence_time)
    matchup = f"{away} @ {home}"
    h_rec = _team_record(home_team) if home_team else ""
    a_rec = _team_record(away_team) if away_team else ""

    fg_spread = float(pred.fg_spread or 0)
    fg_total = float(pred.fg_total or 0)
    fg_prob = float(pred.fg_home_ml_prob or 0.5)
    h1_spread = float(getattr(pred, "h1_spread", 0) or 0)
    h1_total = float(getattr(pred, "h1_total", 0) or 0)

    # Raw model scores for display
    pred_home_fg = float(getattr(pred, "predicted_home_fg", 0) or 0)
    pred_away_fg = float(getattr(pred, "predicted_away_fg", 0) or 0)
    pred_home_1h = float(getattr(pred, "predicted_home_1h", 0) or 0)
    pred_away_1h = float(getattr(pred, "predicted_away_1h", 0) or 0)
    fg_scores = (
        f"{pred_home_fg:.0f}-{pred_away_fg:.0f}"
        if pred_home_fg
        else f"{fg_total/2+fg_spread/2:.0f}-{fg_total/2-fg_spread/2:.0f}"
    )
    h1_scores = (
        f"{pred_home_1h:.0f}-{pred_away_1h:.0f}"
        if pred_home_1h
        else f"{h1_total/2+h1_spread/2:.0f}-{h1_total/2-h1_spread/2:.0f}"
    )

    opening_spread = getattr(pred, "opening_spread", None)
    opening_total = getattr(pred, "opening_total", None)

    picks: list[Pick] = []

    # ── Full-game spread ──────────────────────────────────────
    # opening_spread uses betting convention (negative = home fav).
    # fg_spread = home_score - away_score (positive = home wins).
    # Edge on home side = mkt_spread + fg_spread:
    #   positive → model says home is better than market gives credit
    #   negative → model says away is better than market gives credit
    mkt_spread = (
        float(opening_spread)
        if opening_spread is not None and abs(float(opening_spread)) >= 0.5
        else None
    )
    if mkt_spread is not None:
        home_edge = mkt_spread + fg_spread
        if abs(home_edge) >= min_edge:
            label = f"{home} {mkt_spread:+.1f}" if home_edge > 0 else f"{away} {-mkt_spread:+.1f}"
            e = round(abs(home_edge), 1)
            picks.append(
                Pick(
                    label,
                    e,
                    t_cst,
                    matchup,
                    "FG",
                    "SPREAD",
                    f"{mkt_spread:+.1f}",
                    fg_scores,
                    h_rec,
                    a_rec,
                    _fire_count(e),
                    odds=odds_map.get("FG_SPREAD", ""),
                )
            )
    elif abs(fg_spread) >= min_edge:
        # No market line — convert model margin to betting convention
        label = f"{home} {-fg_spread:+.1f}" if fg_spread > 0 else f"{away} {fg_spread:+.1f}"
        e = round(abs(fg_spread), 1)
        picks.append(
            Pick(
                label,
                e,
                t_cst,
                matchup,
                "FG",
                "SPREAD",
                f"{-fg_spread:+.1f}",
                fg_scores,
                h_rec,
                a_rec,
                _fire_count(e),
                odds=odds_map.get("FG_SPREAD", ""),
            )
        )

    # ── Full-game total ───────────────────────────────────────
    mkt_total = float(opening_total) if opening_total is not None else None
    if mkt_total is not None:
        total_edge = abs(fg_total - mkt_total)
        if total_edge >= min_edge:
            direction = "OVER" if fg_total > mkt_total else "UNDER"
            e = round(total_edge, 1)
            picks.append(
                Pick(
                    f"{direction} {mkt_total:.1f}",
                    e,
                    t_cst,
                    matchup,
                    "FG",
                    "TOTAL",
                    f"{mkt_total:.1f}",
                    fg_scores,
                    h_rec,
                    a_rec,
                    _fire_count(e),
                    odds=odds_map.get("FG_TOTAL", ""),
                )
            )
    else:
        total_diff = abs(fg_total - _NBA_AVG_TOTAL)
        if total_diff >= 5:
            direction = "OVER" if fg_total > _NBA_AVG_TOTAL else "UNDER"
            e = round(total_diff, 1)
            picks.append(
                Pick(
                    f"{direction} {fg_total:.1f}",
                    e,
                    t_cst,
                    matchup,
                    "FG",
                    "TOTAL",
                    f"{fg_total:.1f}",
                    fg_scores,
                    h_rec,
                    a_rec,
                    _fire_count(e),
                    odds=odds_map.get("FG_TOTAL", ""),
                )
            )

    # ── 1H spread ─────────────────────────────────────────────
    if abs(h1_spread) >= min_edge:
        label = f"{home} {-h1_spread:+.1f}" if h1_spread > 0 else f"{away} {h1_spread:+.1f}"
        e = round(abs(h1_spread), 1)
        picks.append(
            Pick(
                label,
                e,
                t_cst,
                matchup,
                "1H",
                "SPREAD",
                f"{-h1_spread:+.1f}",
                h1_scores,
                h_rec,
                a_rec,
                _fire_count(e),
                odds=odds_map.get("1H_SPREAD", ""),
            )
        )

    # ── 1H total ──────────────────────────────────────────────
    h1_total_diff = abs(h1_total - _NBA_AVG_TOTAL / 2)
    if h1_total_diff >= 4:
        direction = "OVER" if h1_total > _NBA_AVG_TOTAL / 2 else "UNDER"
        e = round(h1_total_diff, 1)
        picks.append(
            Pick(
                f"{direction} {h1_total:.1f}",
                e,
                t_cst,
                matchup,
                "1H",
                "TOTAL",
                f"{h1_total:.1f}",
                h1_scores,
                h_rec,
                a_rec,
                _fire_count(e),
                odds=odds_map.get("1H_TOTAL", ""),
            )
        )

    # ── ML pick ────────────────────────────────────────────────
    ml_pts_edge = round(abs(fg_spread), 1)
    if ml_pts_edge >= min_edge:
        side = home if fg_spread > 0 else away
        prob_display = fg_prob if fg_prob >= 0.5 else 1 - fg_prob
        picks.append(
            Pick(
                f"{side} ML",
                ml_pts_edge,
                t_cst,
                matchup,
                "FG",
                "ML",
                f"{prob_display:.0%}",
                fg_scores,
                h_rec,
                a_rec,
                _fire_count(ml_pts_edge),
                odds=odds_map.get("FG_ML", ""),
            )
        )

    return picks


# ── Adaptive Card builder (pick-based, mobile-first) ──────────────


def _pick_row(pick: Pick) -> dict:
    """Build a ColumnSet for a single pick row.

    Layout (mobile-friendly):
      Left (stretch): pick label + fires, matchup with records, segment/market/model
      Right (auto): edge value coloured
    """
    fires = _fire_emojis(pick.edge)
    # Matchup line with records if available
    if pick.away_record and pick.home_record:
        matchup_line = f"{pick.matchup.split(' @ ')[0]} ({pick.away_record}) @ {pick.matchup.split(' @ ')[1]} ({pick.home_record})"
    else:
        matchup_line = pick.matchup
    detail_line = f"{pick.segment} {pick.market_type} · Line: {pick.market_line} · Model: {pick.model_scores}"

    return {
        "type": "ColumnSet",
        "separator": True,
        "spacing": "Medium",
        "columns": [
            {
                "type": "Column",
                "width": "stretch",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"**{pick.label}**{fires}",
                        "wrap": True,
                        "weight": "Bolder",
                    },
                    {
                        "type": "TextBlock",
                        "text": f"{pick.time_cst} · {matchup_line}",
                        "size": "Small",
                        "isSubtle": True,
                        "spacing": "None",
                        "wrap": True,
                    },
                    {
                        "type": "TextBlock",
                        "text": detail_line,
                        "size": "Small",
                        "isSubtle": True,
                        "spacing": "None",
                        "wrap": True,
                    },
                ],
            },
            {
                "type": "Column",
                "width": "auto",
                "verticalContentAlignment": "Center",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"**{pick.edge:.1f}**",
                        "size": "Large",
                        "weight": "Bolder",
                        "color": _edge_color(pick.edge),
                        "horizontalAlignment": "Right",
                    }
                ],
            },
        ],
    }


def build_teams_card(
    predictions_with_games: list[tuple[Any, Any]],
    max_games: int,
    odds_pulled_at: datetime | None = None,
    min_edge: float = MIN_EDGE,
    download_url: str | None = None,
) -> dict[str, Any]:
    """Build a pick-focused Adaptive Card sorted by edge.

    Each row shows: pick label + fires, matchup with records,
    segment/market/model scores, and edge value.
    """
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    odds_ts = (
        odds_pulled_at.strftime("%Y-%m-%d %H:%M UTC") if odds_pulled_at else now_str
    )

    # Extract picks from every prediction
    all_picks: list[Pick] = []
    game_ids: set[int] = set()
    for pred, game in predictions_with_games:
        game_ids.add(getattr(game, "id", id(game)))
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge))

    # Sort by edge descending
    all_picks.sort(key=lambda p: -p.edge)

    n_games = len(game_ids)
    total_picks = len(all_picks)
    max_picks = max_games * 2
    display_picks = all_picks[:max_picks]
    remaining = total_picks - len(display_picks)

    body: list[dict] = [
        {
            "type": "TextBlock",
            "text": "🏀 **NBA Daily Slate**",
            "size": "Large",
            "weight": "Bolder",
            "wrap": True,
        },
        {
            "type": "TextBlock",
            "text": f"Model {MODEL_VERSION} · Odds pulled {odds_ts}",
            "wrap": True,
            "size": "Small",
            "isSubtle": True,
            "spacing": "None",
        },
        {
            "type": "TextBlock",
            "text": f"{n_games} games · {total_picks} qualified picks",
            "wrap": True,
            "size": "Medium",
            "weight": "Bolder",
            "spacing": "Small",
        },
        {
            "type": "ColumnSet",
            "spacing": "Small",
            "columns": [
                {
                    "type": "Column",
                    "width": "stretch",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": "**PICK**",
                            "size": "Small",
                            "isSubtle": True,
                        }
                    ],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": "**EDGE**",
                            "size": "Small",
                            "isSubtle": True,
                            "horizontalAlignment": "Right",
                        }
                    ],
                },
            ],
        },
    ]

    for pick in display_picks:
        body.append(_pick_row(pick))

    if remaining > 0:
        body.append(
            {
                "type": "TextBlock",
                "text": f"• {remaining} more picks — download full slate below",
                "size": "Small",
                "isSubtle": True,
                "spacing": "Medium",
                "wrap": True,
            }
        )

    card: dict[str, Any] = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }

    if download_url:
        card["actions"] = [
            {
                "type": "Action.OpenUrl",
                "title": "📥 Download Full Slate (CSV)",
                "url": download_url,
            }
        ]

    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card,
            }
        ],
    }


# ── CSV slate builder ──────────────────────────────────────────────


def build_slate_csv(
    predictions_with_games: list[tuple[Any, ...]],
    min_edge: float = MIN_EDGE,
) -> str:
    """Build a CSV string of the full slate with all columns.

    Columns: Time (CT),Matchup,Home Record,Away Record,Segment,Market,
             Line,Pick,Odds,Model Scores,Edge,Rating

    Each element may be ``(pred, game)`` or ``(pred, game, odds_map)``.
    """
    all_picks: list[Pick] = []
    for row in predictions_with_games:
        pred, game = row[0], row[1]
        om: dict[str, str] = row[2] if len(row) > 2 else {}  # type: ignore[index]
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge, odds_map=om))
    all_picks.sort(key=lambda p: -p.edge)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Time (CT)",
            "Matchup",
            "Home Record",
            "Away Record",
            "Segment",
            "Market",
            "Line",
            "Pick",
            "Odds",
            "Model Scores",
            "Edge",
            "Rating",
        ]
    )
    for p in all_picks:
        writer.writerow(
            [
                p.time_cst,
                p.matchup,
                p.home_record,
                p.away_record,
                p.segment,
                p.market_type,
                p.market_line,
                p.label,
                p.odds,
                p.model_scores,
                p.edge,
                "\U0001f525" * p.confidence,
            ]
        )
    return buf.getvalue()


# ── HTML slate builder ─────────────────────────────────────────────


def _esc(val: object) -> str:
    return html_mod.escape(str(val))


def _edge_css_color(edge: float) -> str:
    if edge >= 7:
        return "#16a34a"  # green — hot
    if edge >= 5:
        return "#ca8a04"  # amber — warm
    if edge >= 3:
        return "#2563eb"  # blue — mild
    return "#6b7280"  # gray — flat


def _confidence_badge(fires: int) -> str:
    """Inline-styled confidence badge from fire count (1-5)."""
    if fires >= 4:
        bg, fg = "#dcfce7", "#15803d"  # green
    elif fires >= 3:
        bg, fg = "#fef9c3", "#854d0e"  # amber
    else:
        bg, fg = "#fef2f2", "#dc2626"  # red
    label = "\U0001f525" * fires
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 6px;'
        f'border-radius:4px;font-weight:700;font-size:13px">{label}</span>'
    )


def _segment_pill(seg: str) -> str:
    if seg == "FG":
        bg, fg, bd = "rgba(52,168,83,.15)", "#34a853", "rgba(52,168,83,.3)"
    else:
        bg, fg, bd = "rgba(66,133,244,.15)", "#4285F4", "rgba(66,133,244,.3)"
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {bd};'
        f"padding:2px 6px;border-radius:4px;font-weight:600;font-size:11px;"
        f'text-transform:uppercase;letter-spacing:.5px">{_esc(seg)}</span>'
    )


def _pick_side_border(pick: Pick) -> str:
    """Return left-border colour: green for home/over, blue for away/under."""
    label_lower = pick.label.lower()
    if pick.market_type == "TOTAL":
        return "#16a34a" if "over" in label_lower else "#2563eb"
    matchup_parts = pick.matchup.split(" @ ")
    home_name = matchup_parts[1] if len(matchup_parts) == 2 else ""
    if home_name and pick.label.startswith(home_name):
        return "#16a34a"  # home = green
    return "#2563eb"  # away = blue


def build_html_slate(
    predictions_with_games: list[tuple[Any, ...]],
    odds_pulled_at: datetime | None = None,
    min_edge: float = MIN_EDGE,
) -> str:
    """Build a styled HTML table of the full slate for posting directly to Teams.

    Teams HTML messages support inline CSS — no JS, no external fonts.
    Formatting draws from NCAAM v2.0 / NBA v5.0 visual patterns.

    Each element may be ``(pred, game)`` or ``(pred, game, odds_map)``.
    """
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    odds_ts = (
        odds_pulled_at.strftime("%Y-%m-%d %H:%M UTC") if odds_pulled_at else now_str
    )
    today_display = datetime.now(_CST).strftime("%A, %B %d, %Y")

    all_picks: list[Pick] = []
    game_ids: set[int] = set()
    for row in predictions_with_games:
        pred, game = row[0], row[1]
        om: dict[str, str] = row[2] if len(row) > 2 else {}  # type: ignore[index]
        game_ids.add(getattr(game, "id", id(game)))
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge, odds_map=om))
    all_picks.sort(key=lambda p: -p.edge)

    n_games = len(game_ids)

    # ── header ─────────────────────────────────────
    header = (
        '<div style="margin-bottom:12px">'
        '<div style="font-size:20px;font-weight:700;color:#1a2332">'
        "\U0001f3c0 NBA Daily Slate</div>"
        f'<div style="font-size:13px;color:#6b7280">{_esc(today_display)}</div>'
        f'<div style="font-size:12px;color:#6b7280">'
        f"Model {_esc(MODEL_VERSION)} &middot; Odds pulled {_esc(odds_ts)} "
        f"&middot; {n_games} games &middot; {len(all_picks)} picks</div>"
        "</div>"
    )

    if not all_picks:
        return header + '<p style="color:#6b7280">No qualified picks today.</p>'

    # ── table ──────────────────────────────────────
    th_style = (
        "background:#1a2332;color:#d4af37;font-weight:600;font-size:11px;"
        "letter-spacing:1px;text-transform:uppercase;padding:8px 6px;"
        "text-align:left;white-space:nowrap"
    )

    rows_html: list[str] = []
    for i, p in enumerate(all_picks):
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fa"
        border_color = _pick_side_border(p)
        edge_color = _edge_css_color(p.edge)

        # Matchup with records
        parts = p.matchup.split(" @ ")
        away_display = f'<span style="color:#2563eb">{_esc(parts[0])}</span>'
        if p.away_record:
            away_display += f' <span style="font-size:11px;color:#9ca3af">({_esc(p.away_record)})</span>'
        home_display = f'<span style="color:#16a34a;font-weight:700">{_esc(parts[1]) if len(parts) > 1 else ""}</span>'
        if p.home_record:
            home_display += f' <span style="font-size:11px;color:#9ca3af">({_esc(p.home_record)})</span>'
        matchup_cell = f"{away_display} @ {home_display}"

        td = 'style="padding:6px;border-bottom:1px solid #e9ecef;font-size:13px;vertical-align:middle"'

        # Odds badge (shown after pick label)
        odds_html = ""
        if p.odds:
            odds_fg = "#16a34a" if p.odds.startswith("+") else "#dc2626"
            odds_html = (
                f' <span style="font-size:12px;font-weight:700;color:{odds_fg};'
                f'background:rgba(0,0,0,.04);padding:1px 4px;border-radius:3px">'
                f"({_esc(p.odds)})</span>"
            )

        rows_html.append(
            f'<tr style="background:{bg}">'
            f"<td {td}>{_esc(p.time_cst)}</td>"
            f"<td {td}>{matchup_cell}</td>"
            f"<td {td}>{_segment_pill(p.segment)}</td>"
            f'<td {td}><span style="font-weight:600;color:#374151;font-size:12px">{_esc(p.market_type)}</span></td>'
            f'<td style="padding:6px;border-bottom:1px solid #e9ecef;font-size:13px;'
            f'vertical-align:middle;border-left:3px solid {border_color};font-weight:700">'
            f"{_esc(p.label)}{odds_html}"
            f' <span style="font-size:11px;color:#9ca3af;font-weight:400">{_esc(p.market_line)}</span></td>'
            f'<td {td}><span style="font-weight:600">{_esc(p.model_scores)}</span></td>'
            f'<td style="padding:6px;border-bottom:1px solid #e9ecef;text-align:center;'
            f'font-weight:700;font-size:14px;color:{edge_color}">{p.edge:.1f}</td>'
            f"<td {td}>{_confidence_badge(p.confidence)}</td>"
            "</tr>"
        )

    table = (
        '<table style="width:100%;border-collapse:collapse;border:1px solid #dee2e6">'
        "<thead><tr>"
        f'<th style="{th_style}">Time</th>'
        f'<th style="{th_style}">Matchup</th>'
        f'<th style="{th_style}">Seg</th>'
        f'<th style="{th_style}">Market</th>'
        f'<th style="{th_style}">Pick</th>'
        f'<th style="{th_style}">Model</th>'
        f'<th style="{th_style};text-align:center">Edge</th>'
        f'<th style="{th_style};text-align:center">Rating</th>'
        "</tr></thead>"
        f'<tbody>{"".join(rows_html)}</tbody>'
        "</table>"
    )

    footer = (
        f'<div style="text-align:center;padding:8px;color:#9ca3af;font-size:11px">'
        f"GBSV NBA {_esc(MODEL_VERSION)} &middot; Sorted by edge descending</div>"
    )

    return header + table + footer


# ── Legacy plain-text builder (kept for backward compat / tests) ──


def _format_game_line(pred: Any, game: Any) -> str:
    home_name = (
        game.home_team.name
        if game.home_team is not None
        else f"Team {game.home_team_id}"
    )
    away_name = (
        game.away_team.name
        if game.away_team is not None
        else f"Team {game.away_team_id}"
    )
    kickoff = game.commence_time
    kickoff_text = (
        kickoff.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC") if kickoff else "TBD"
    )

    fg_prob = float(pred.fg_home_ml_prob or 0.5)
    away_prob = 1.0 - fg_prob
    return (
        f"- {away_name} at {home_name} ({kickoff_text})\n"
        f"  FG spread {float(pred.fg_spread or 0.0):+.1f} | "
        f"FG total {float(pred.fg_total or 0.0):.1f} | "
        f"FG ML H {fg_prob:.3f} / A {away_prob:.3f}"
    )


def build_teams_text(
    predictions_with_games: list[tuple[Any, Any]], max_games: int
) -> str:
    lines = ["NBA Predictions Update", ""]
    lines.append(f"Games: {min(len(predictions_with_games), max_games)}")
    lines.append("")

    for pred, game in predictions_with_games[:max_games]:
        lines.append(_format_game_line(pred, game))

    return "\n".join(lines)


# ── Senders ────────────────────────────────────────────────────────


async def send_card_to_teams(webhook_url: str, payload: dict[str, Any]) -> None:
    """Send an Adaptive Card payload to a Teams incoming webhook."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(webhook_url, json=payload)
        response.raise_for_status()


async def send_text_to_teams(webhook_url: str, text: str) -> None:
    payload: dict[str, Any] = {"text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(webhook_url, json=payload)
        response.raise_for_status()


async def send_card_via_graph(
    team_id: str, channel_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Post an Adaptive Card to a Teams channel via Microsoft Graph API.

    Uses ``DefaultAzureCredential`` to obtain a token (works with
    Azure CLI, managed identity, environment creds, etc.).

    ``payload`` should be the dict returned by ``build_teams_card()``.
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")

    # Extract the Adaptive Card from the webhook-format payload
    attachment = payload["attachments"][0]
    card_content = attachment["content"]

    graph_body = {
        "body": {
            "contentType": "html",
            "content": '<attachment id="card-1"></attachment>',
        },
        "attachments": [
            {
                "id": "card-1",
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": json.dumps(card_content, ensure_ascii=False),
            }
        ],
    }

    url = (
        f"https://graph.microsoft.com/v1.0/teams/{team_id}"
        f"/channels/{channel_id}/messages"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json=graph_body,
            headers={
                "Authorization": f"Bearer {token.token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()


async def send_html_via_graph(
    team_id: str, channel_id: str, html_content: str
) -> dict[str, Any]:
    """Post an HTML message directly to a Teams channel via Microsoft Graph API.

    Renders as native HTML in Teams — no Adaptive Card, no login prompt.
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")

    graph_body = {
        "body": {
            "contentType": "html",
            "content": html_content,
        },
    }

    url = (
        f"https://graph.microsoft.com/v1.0/teams/{team_id}"
        f"/channels/{channel_id}/messages"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json=graph_body,
            headers={
                "Authorization": f"Bearer {token.token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()


async def upload_csv_to_channel(
    team_id: str,
    channel_id: str,
    filename: str,
    csv_content: str,
) -> str:
    """Upload a CSV file to the Teams channel's Files tab and return the web URL.

    Uses the Graph API to:
    1. Get the channel's filesFolder (SharePoint drive)
    2. PUT the CSV content to that folder
    3. Return the webUrl for the uploaded file
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")
    headers = {
        "Authorization": f"Bearer {token.token}",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get the channel's files folder (SharePoint drive item)
        folder_url = (
            f"https://graph.microsoft.com/v1.0/teams/{team_id}"
            f"/channels/{channel_id}/filesFolder"
        )
        folder_resp = await client.get(folder_url, headers=headers)
        folder_resp.raise_for_status()
        folder_data = folder_resp.json()

        drive_id = folder_data["parentReference"]["driveId"]
        folder_id = folder_data["id"]

        # Upload the CSV file to the channel's files folder
        upload_url = (
            f"https://graph.microsoft.com/v1.0/drives/{drive_id}"
            f"/items/{folder_id}:/{filename}:/content"
        )
        upload_resp = await client.put(
            upload_url,
            content=csv_content.encode("utf-8-sig"),
            headers={
                **headers,
                "Content-Type": "text/csv",
            },
        )
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()

        web_url: str = upload_data.get("webUrl", "")
        logger.info("Uploaded %s → %s", filename, web_url)
        return web_url
