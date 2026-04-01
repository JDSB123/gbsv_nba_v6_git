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

from src.config import get_nba_avg_total
from src.config import get_settings as _get_settings
from src.models.versioning import MODEL_VERSION

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


# ── Helpers ────────────────────────────────────────────────────────


def _get_model_modified_at() -> str:
    """Timestamp of the newest model artifact file."""
    try:
        latest = max(
            (f.stat().st_mtime for f in _ARTIFACTS_DIR.glob("model_*.json") if f.exists()),
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


# Minimum edge (in points) for a pick to qualify — from central config
MIN_EDGE = _get_settings().min_edge
EDGE_THRESHOLDS = _get_settings().edge_thresholds

_get_nba_avg_total = get_nba_avg_total  # backward-compat alias


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
    """Format a datetime as 'YYYY-MM-DD 6:30 PM CT' for clarity and sortability."""
    if not dt:
        return "TBD"
    # commence_time is stored as naive UTC; make it aware before converting
    aware = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    ct = aware.astimezone(_CST)
    raw_time = ct.strftime("%I:%M %p").lstrip("0")
    return f"{ct.strftime('%Y-%m-%d')} {raw_time} CT"


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
    market_line: str  # "-3.5", "222.5", "-160"
    model_scores: str  # "Celtics 112, Heat 108"
    home_record: str  # "42-18" or ""
    away_record: str  # "28-32" or ""
    confidence: int  # 1-5 fire count
    odds: str = ""  # American odds, e.g. "-110", "+150"
    rationale: str = ""  # Brief explanation of the edge


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


def _prob_to_american(prob: float) -> str:
    """Convert win probability (0-1) to American odds string."""
    if prob <= 0.01 or prob >= 0.99:
        return ""
    if prob >= 0.5:
        return f"{int(round(-(prob / (1 - prob)) * 100)):+d}"
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def _american_to_prob(odds_str: str) -> float | None:
    """Convert American odds string to implied probability (0-1)."""
    if not odds_str:
        return None
    try:
        odds = float(odds_str.replace("+", ""))
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
    except ValueError:
        return None


def _consensus_price(books: dict, key: str) -> str:
    """Average a numeric price field across all books, return American odds string."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    if not vals:
        return ""
    return f"{int(round(sum(vals) / len(vals))):+d}"


def _consensus_line(books: dict, key: str) -> float | None:
    """Average a numeric line/point field across all books."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def extract_picks(
    pred: Any,
    game: Any,
    min_edge: float = MIN_EDGE,
    odds_map: dict[str, str] | None = None,
) -> list[Pick]:
    """Extract actionable picks with edge values from a prediction.

    Each pick includes segment, market line, model scores with team names,
    American odds/price, confidence rating, and a brief rationale.
    """
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

    # Raw model scores for display — include team names
    pred_home_fg = float(getattr(pred, "predicted_home_fg", 0) or 0)
    pred_away_fg = float(getattr(pred, "predicted_away_fg", 0) or 0)
    pred_home_1h = float(getattr(pred, "predicted_home_1h", 0) or 0)
    pred_away_1h = float(getattr(pred, "predicted_away_1h", 0) or 0)
    if pred_home_fg:
        fg_scores = f"{home} {pred_home_fg:.0f}, {away} {pred_away_fg:.0f}"
    else:
        h = fg_total / 2 + fg_spread / 2
        a = fg_total / 2 - fg_spread / 2
        fg_scores = f"{home} {h:.0f}, {away} {a:.0f}"
    if pred_home_1h:
        h1_scores = f"{home} {pred_home_1h:.0f}, {away} {pred_away_1h:.0f}"
    else:
        h = h1_total / 2 + h1_spread / 2
        a = h1_total / 2 - h1_spread / 2
        h1_scores = f"{home} {h:.0f}, {away} {a:.0f}"

    opening_spread = getattr(pred, "opening_spread", None)
    opening_total = getattr(pred, "opening_total", None)

    # Extract 1H opening lines and consensus odds from stored odds detail
    odds_sourced = getattr(pred, "odds_sourced", None) or {}
    books = odds_sourced.get("books", {}) if isinstance(odds_sourced, dict) else {}
    opening_h1_spread = (
        odds_sourced.get("opening_h1_spread") if isinstance(odds_sourced, dict) else None
    )
    opening_h1_total = (
        odds_sourced.get("opening_h1_total") if isinstance(odds_sourced, dict) else None
    )

    # Build consensus American odds from per-book data if not supplied
    if not odds_map:
        odds_map = {}
        if books:
            sp = _consensus_price(books, "spread_price")
            if sp:
                odds_map["FG_SPREAD"] = sp
            tp = _consensus_price(books, "total_price")
            if tp:
                odds_map["FG_TOTAL"] = tp
            hml = _consensus_price(books, "home_ml")
            if hml:
                odds_map["FG_ML_HOME"] = hml
            aml = _consensus_price(books, "away_ml")
            if aml:
                odds_map["FG_ML_AWAY"] = aml
            sp1h = _consensus_price(books, "spread_h1_price")
            if sp1h:
                odds_map["1H_SPREAD"] = sp1h
            tp1h = _consensus_price(books, "total_h1_price")
            if tp1h:
                odds_map["1H_TOTAL"] = tp1h
            hml1h = _consensus_price(books, "home_ml_h1")
            if hml1h:
                odds_map["1H_ML_HOME"] = hml1h
            aml1h = _consensus_price(books, "away_ml_h1")
            if aml1h:
                odds_map["1H_ML_AWAY"] = aml1h

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
            pick_home = home_edge > 0
            label = f"{home} {mkt_spread:+.1f}" if pick_home else f"{away} {-mkt_spread:+.1f}"
            e = round(abs(home_edge), 1)
            side_name = home if pick_home else away
            rationale = (
                f"Model: {home} by {fg_spread:+.1f} vs line {mkt_spread:+.1f} → "
                f"{e:.1f}pt edge on {side_name}"
            )
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
                    rationale=rationale,
                )
            )
    elif abs(fg_spread) >= min_edge:
        # No market line — convert model margin to betting convention
        pick_home = fg_spread > 0
        label = f"{home} {-fg_spread:+.1f}" if pick_home else f"{away} {fg_spread:+.1f}"
        e = round(abs(fg_spread), 1)
        side_name = home if pick_home else away
        rationale = f"No market line · Model projects {side_name} by {e:.1f}pts"
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
                rationale=rationale,
            )
        )

    # ── Full-game total ───────────────────────────────────────
    mkt_total = float(opening_total) if opening_total is not None else None
    if mkt_total is not None:
        total_edge = abs(fg_total - mkt_total)
        if total_edge >= min_edge:
            direction = "OVER" if fg_total > mkt_total else "UNDER"
            e = round(total_edge, 1)
            rationale = (
                f"Model total {fg_total:.1f} vs line {mkt_total:.1f} → "
                f"{e:.1f}pt {direction.lower()}"
            )
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
                    rationale=rationale,
                )
            )
    else:
        nba_avg = _get_nba_avg_total()
        total_diff = abs(fg_total - nba_avg)
        if total_diff >= 5:
            direction = "OVER" if fg_total > nba_avg else "UNDER"
            e = round(total_diff, 1)
            rationale = f"Model total {fg_total:.1f} vs NBA avg {nba_avg:.0f} → {e:.1f}pt {direction.lower()}"
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
                    rationale=rationale,
                )
            )

    # ── 1H spread ─────────────────────────────────────────────
    mkt_h1_spread = (
        float(opening_h1_spread)
        if opening_h1_spread is not None and abs(float(opening_h1_spread)) >= 0.5
        else None
    )
    if mkt_h1_spread is not None:
        # Compare model vs market, same formula as FG spread
        h1_home_edge = mkt_h1_spread + h1_spread
        if abs(h1_home_edge) >= min_edge:
            pick_home = h1_home_edge > 0
            label = f"{home} {mkt_h1_spread:+.1f}" if pick_home else f"{away} {-mkt_h1_spread:+.1f}"
            e = round(abs(h1_home_edge), 1)
            side_name = home if pick_home else away
            rationale = (
                f"1H Model: {home} by {h1_spread:+.1f} vs line {mkt_h1_spread:+.1f} → "
                f"{e:.1f}pt edge on {side_name}"
            )
            picks.append(
                Pick(
                    label,
                    e,
                    t_cst,
                    matchup,
                    "1H",
                    "SPREAD",
                    f"{mkt_h1_spread:+.1f}",
                    h1_scores,
                    h_rec,
                    a_rec,
                    _fire_count(e),
                    odds=odds_map.get("1H_SPREAD", ""),
                    rationale=rationale,
                )
            )
    elif abs(h1_spread) >= min_edge:
        # No 1H market line — use model margin directly
        pick_home = h1_spread > 0
        label = f"{home} {-h1_spread:+.1f}" if pick_home else f"{away} {h1_spread:+.1f}"
        e = round(abs(h1_spread), 1)
        side_name = home if pick_home else away
        rationale = f"No 1H line · Model projects {side_name} 1H by {e:.1f}pts"
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
                rationale=rationale,
            )
        )

    # ── 1H total ──────────────────────────────────────────────
    mkt_h1_total = float(opening_h1_total) if opening_h1_total is not None else None
    if mkt_h1_total is not None:
        h1_total_edge = abs(h1_total - mkt_h1_total)
        if h1_total_edge >= min_edge:
            direction = "OVER" if h1_total > mkt_h1_total else "UNDER"
            e = round(h1_total_edge, 1)
            rationale = (
                f"1H Model total {h1_total:.1f} vs line {mkt_h1_total:.1f} → "
                f"{e:.1f}pt {direction.lower()}"
            )
            picks.append(
                Pick(
                    f"{direction} {mkt_h1_total:.1f}",
                    e,
                    t_cst,
                    matchup,
                    "1H",
                    "TOTAL",
                    f"{mkt_h1_total:.1f}",
                    h1_scores,
                    h_rec,
                    a_rec,
                    _fire_count(e),
                    odds=odds_map.get("1H_TOTAL", ""),
                    rationale=rationale,
                )
            )
    else:
        h1_avg = _get_nba_avg_total() / 2
        h1_total_diff = abs(h1_total - h1_avg)
        if h1_total_diff >= 4:
            direction = "OVER" if h1_total > h1_avg else "UNDER"
            e = round(h1_total_diff, 1)
            rationale = (
                f"1H Model total {h1_total:.1f} vs avg {h1_avg:.0f} → {e:.1f}pt {direction.lower()}"
            )
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
                    rationale=rationale,
                )
            )

    # ── ML pick ────────────────────────────────────────────────
    pick_home = fg_spread > 0
    side = home if pick_home else away
    win_prob = fg_prob if pick_home else 1 - fg_prob

    # American odds — prefer book consensus, else convert from probability
    if pick_home:
        ml_odds = odds_map.get("FG_ML_HOME", "") or _prob_to_american(fg_prob)
    else:
        ml_odds = odds_map.get("FG_ML_AWAY", "") or _prob_to_american(1 - fg_prob)

    market_prob = _american_to_prob(ml_odds)

    # Calculate prob edge in percentage points (e.g. 0.05 = 5%).
    # If market_prob is completely unavailable, fall back to comparing against a baseline of 0.5 (coin flip)
    # to still surface strong model beliefs when books haven't posted odds.
    prob_edge = (win_prob - market_prob) if market_prob is not None else (win_prob - 0.5)

    # If the model edge is positive against the market
    if prob_edge > 0.02:  # Min 2% edge
        # Scale to "points" equivalent for fire count (approx 3% per point)
        ml_pts_edge = round(prob_edge * 33.3, 1)

        m_prob_str = f"{market_prob:.0%}" if market_prob is not None else "N/A"
        rationale = (
            f"Model projects {side} by {abs(fg_spread):.1f}pts. "
            f"Edge: {win_prob:.0%} win prob vs {m_prob_str} implied ({ml_odds or 'n/a'})"
        )
        picks.append(
            Pick(
                f"{side} ML",
                ml_pts_edge,
                t_cst,
                matchup,
                "FG",
                "ML",
                ml_odds or _prob_to_american(win_prob),
                fg_scores,
                h_rec,
                a_rec,
                _fire_count(ml_pts_edge),
                odds=ml_odds,
                rationale=rationale,
            )
        )

    return picks


# ── Adaptive Card builder (pick-based, mobile-first) ──────────────


def _pick_row(pick: Pick) -> dict:
    """Build a ColumnSet for a single pick row.

    Layout (mobile-friendly):
      Left (stretch): pick label + odds + fires, matchup with records,
                       segment/market/model scores, rationale
      Right (auto): edge value coloured
    """
    fires = _fire_emojis(pick.edge)
    odds_tag = f" ({pick.odds})" if pick.odds else ""
    # Matchup line with records if available
    if pick.away_record and pick.home_record:
        matchup_line = f"{pick.matchup.split(' @ ')[0]} ({pick.away_record}) @ {pick.matchup.split(' @ ')[1]} ({pick.home_record})"
    else:
        matchup_line = pick.matchup
    detail_line = (
        f"{pick.segment} {pick.market_type} · Line: {pick.market_line} · Model: {pick.model_scores}"
    )

    items: list[dict] = [
        {
            "type": "TextBlock",
            "text": f"**{pick.label}**{odds_tag}{fires}",
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
    ]
    if pick.rationale:
        items.append(
            {
                "type": "TextBlock",
                "text": f"_{pick.rationale}_",
                "size": "Small",
                "isSubtle": True,
                "spacing": "None",
                "wrap": True,
            }
        )

    return {
        "type": "ColumnSet",
        "separator": True,
        "spacing": "Medium",
        "columns": [
            {
                "type": "Column",
                "width": "stretch",
                "items": items,
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


def _odds_source_block(odds_sourced: dict | None) -> list[dict]:
    """Build Adaptive Card elements showing per-book odds breakdown."""
    if not odds_sourced:
        return []
    books = odds_sourced.get("books", {})
    if not books:
        return []
    parts: list[str] = []
    for bk, lines in sorted(books.items()):
        pieces = []
        if "spread" in lines:
            price_str = f" ({lines['spread_price']:+d})" if lines.get("spread_price") else ""
            pieces.append(f"Sprd {lines['spread']:+.1f}{price_str}")
        if "total" in lines:
            price_str = f" ({lines['total_price']:+d})" if lines.get("total_price") else ""
            pieces.append(f"O/U {lines['total']:.1f}{price_str}")
        if "home_ml" in lines:
            pieces.append(f"ML {lines['home_ml']:+d}")
        # 1H markets
        if "spread_h1" in lines:
            price_str = f" ({lines['spread_h1_price']:+d})" if lines.get("spread_h1_price") else ""
            pieces.append(f"1H Sprd {lines['spread_h1']:+.1f}{price_str}")
        if "total_h1" in lines:
            price_str = f" ({lines['total_h1_price']:+d})" if lines.get("total_h1_price") else ""
            pieces.append(f"1H O/U {lines['total_h1']:.1f}{price_str}")
        if pieces:
            parts.append(f"**{bk}**: {' · '.join(pieces)}")
    if not parts:
        return []
    ts_raw = odds_sourced.get("captured_at", "")
    ts_display = ""
    if ts_raw:
        try:
            dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(_CST)
            ts_display = f" (as of {dt.strftime('%I:%M %p').lstrip('0')} CT)"
        except Exception:
            pass
    return [
        {
            "type": "TextBlock",
            "text": f"📊 {' | '.join(parts)}{ts_display}",
            "size": "Small",
            "isSubtle": True,
            "wrap": True,
            "spacing": "None",
        }
    ]


def build_teams_card(
    predictions_with_games: list[tuple[Any, Any]],
    max_games: int,
    odds_pulled_at: datetime | None = None,
    min_edge: float = MIN_EDGE,
    download_url: str | None = None,
    csv_download_url: str | None = None,
) -> dict[str, Any]:
    """Build a pick-focused Adaptive Card sorted by edge.

    Each row shows: pick label + fires, matchup with records,
    segment/market/model scores, and edge value.
    """
    now_cst = datetime.now(_CST)
    now_str = now_cst.strftime("%Y-%m-%d %I:%M %p CT")
    odds_ts = (
        odds_pulled_at.astimezone(_CST).strftime("%Y-%m-%d %I:%M %p CT")
        if odds_pulled_at
        else now_str
    )

    # Extract picks from every prediction
    all_picks: list[Pick] = []
    game_ids: set[int] = set()
    odds_by_game: dict[int, dict] = {}
    game_labels: dict[int, str] = {}
    for pred, game in predictions_with_games:
        gid = getattr(game, "id", id(game))
        game_ids.add(gid)
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge))
        sourced = getattr(pred, "odds_sourced", None)
        if sourced:
            odds_by_game[gid] = sourced
        home = game.home_team.name if game.home_team else "Home"
        away = game.away_team.name if game.away_team else "Away"
        game_labels[gid] = f"{away} @ {home}"

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

    # ── Per-game odds source breakdown ────────────────────────
    if odds_by_game:
        body.append(
            {
                "type": "TextBlock",
                "text": "📊 **Odds Sources**",
                "size": "Small",
                "weight": "Bolder",
                "spacing": "Large",
                "separator": True,
            }
        )
        for gid, detail in odds_by_game.items():
            label = game_labels.get(gid, "")
            blocks = _odds_source_block(detail)
            if blocks and label:
                body.append(
                    {
                        "type": "TextBlock",
                        "text": f"**{label}**",
                        "size": "Small",
                        "spacing": "Small",
                        "wrap": True,
                    }
                )
                body.extend(blocks)

    card: dict[str, Any] = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }

    actions: list[dict[str, str]] = []
    if download_url:
        actions.append(
            {
                "type": "Action.OpenUrl",
                "title": "📊 View Full Slate",
                "url": download_url,
            }
        )
    if csv_download_url:
        actions.append(
            {
                "type": "Action.OpenUrl",
                "title": "📥 Download CSV",
                "url": csv_download_url,
            }
        )
    if actions:
        card["actions"] = actions

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
             Line,Pick,Odds,Model Scores,Edge,Rating,Odds Source,Odds Pulled

    Each element may be ``(pred, game)`` or ``(pred, game, odds_map)``.
    """
    all_picks: list[Pick] = []
    # Build matchup → odds_sourced lookup
    odds_by_matchup: dict[str, dict] = {}
    for row in predictions_with_games:
        pred, game = row[0], row[1]
        om: dict[str, str] = row[2] if len(row) > 2 else {}  # type: ignore[index]
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge, odds_map=om))
        sourced = getattr(pred, "odds_sourced", None)
        if sourced:
            home = game.home_team.name if game.home_team else "Home"
            away = game.away_team.name if game.away_team else "Away"
            odds_by_matchup[f"{away} @ {home}"] = sourced
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
            "Rationale",
            "Odds Source",
            "Odds Pulled",
        ]
    )
    for p in all_picks:
        sourced = odds_by_matchup.get(p.matchup, {})
        books = sourced.get("books", {})
        source_parts = []
        for bk, lines in sorted(books.items()):
            pieces = []
            if "spread" in lines:
                pieces.append(f"S:{lines['spread']:+.1f}")
            if "total" in lines:
                pieces.append(f"T:{lines['total']:.1f}")
            if pieces:
                source_parts.append(f"{bk}({'/'.join(pieces)})")
        odds_source_str = "; ".join(source_parts)
        odds_pulled_str = sourced.get("captured_at", "")
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
                p.rationale,
                odds_source_str,
                odds_pulled_str,
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
    empty_message: str | None = None,
) -> str:
    """Build a styled HTML table of the full slate for posting directly to Teams.

    Teams HTML messages support inline CSS — no JS, no external fonts.
    Formatting draws from NCAAM v2.0 / NBA v5.0 visual patterns.

    Each element may be ``(pred, game)`` or ``(pred, game, odds_map)``.
    """
    now_cst = datetime.now(_CST)
    now_str = now_cst.strftime("%Y-%m-%d %I:%M %p CT")
    odds_ts = (
        odds_pulled_at.astimezone(_CST).strftime("%Y-%m-%d %I:%M %p CT")
        if odds_pulled_at
        else now_str
    )
    today_display = now_cst.strftime("%A, %B %d, %Y")

    all_picks: list[Pick] = []
    game_ids: set[int] = set()
    odds_by_game: dict[int, dict] = {}
    game_labels: dict[int, str] = {}
    for row in predictions_with_games:
        pred, game = row[0], row[1]
        om: dict[str, str] = row[2] if len(row) > 2 else {}  # type: ignore[index]
        gid = getattr(game, "id", id(game))
        game_ids.add(gid)
        all_picks.extend(extract_picks(pred, game, min_edge=min_edge, odds_map=om))
        sourced = getattr(pred, "odds_sourced", None)
        if sourced:
            odds_by_game[gid] = sourced
        home = game.home_team.name if game.home_team else "Home"
        away = game.away_team.name if game.away_team else "Away"
        game_labels[gid] = f"{away} @ {home}"
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
        message = empty_message or "No qualified picks today."
        return header + f'<p style="color:#6b7280">{_esc(message)}</p>'

    # Collect unique values for filter dropdowns
    matchups = sorted({p.matchup for p in all_picks})
    segments = sorted({p.segment for p in all_picks})
    markets = sorted({p.market_type for p in all_picks})

    # ── filter bar ─────────────────────────────────
    sel_style = (
        "padding:6px 10px;border:1px solid #dee2e6;border-radius:6px;"
        "font-size:13px;background:#fff;color:#1a2332;cursor:pointer"
    )
    matchup_opts = "".join(f'<option value="{_esc(m)}">{_esc(m)}</option>' for m in matchups)
    seg_opts = "".join(f'<option value="{_esc(s)}">{_esc(s)}</option>' for s in segments)
    mkt_opts = "".join(f'<option value="{_esc(m)}">{_esc(m)}</option>' for m in markets)
    filter_bar = (
        '<div id="filters" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:center">'
        f'<select id="fMatchup" style="{sel_style}" onchange="applyFilters()">'
        f'<option value="">All Matchups</option>{matchup_opts}</select>'
        f'<select id="fSeg" style="{sel_style}" onchange="applyFilters()">'
        f'<option value="">All Segments</option>{seg_opts}</select>'
        f'<select id="fMkt" style="{sel_style}" onchange="applyFilters()">'
        f'<option value="">All Markets</option>{mkt_opts}</select>'
        f'<input id="fEdge" type="number" step="0.5" min="0" placeholder="Min Edge" '
        f'style="{sel_style};width:100px" oninput="applyFilters()">'
        '<button onclick="resetFilters()" style="padding:6px 14px;border:none;'
        "border-radius:6px;background:#1a2332;color:#d4af37;font-weight:600;"
        'font-size:12px;cursor:pointer;letter-spacing:.5px">RESET</button>'
        "</div>"
    )

    # ── table ──────────────────────────────────────
    th_style = (
        "background:#1a2332;color:#d4af37;font-weight:600;font-size:11px;"
        "letter-spacing:1px;text-transform:uppercase;padding:8px 6px;"
        "text-align:left;white-space:nowrap;cursor:pointer;user-select:none"
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
            away_display += (
                f' <span style="font-size:11px;color:#9ca3af">({_esc(p.away_record)})</span>'
            )
        home_display = f'<span style="color:#16a34a;font-weight:700">{_esc(parts[1]) if len(parts) > 1 else ""}</span>'
        if p.home_record:
            home_display += (
                f' <span style="font-size:11px;color:#9ca3af">({_esc(p.home_record)})</span>'
            )
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

        rationale_html = ""
        if p.rationale:
            rationale_html = (
                f'<div style="font-size:11px;color:#6b7280;font-style:italic;'
                f'margin-top:2px">{_esc(p.rationale)}</div>'
            )

        rows_html.append(
            f'<tr style="background:{bg}" data-matchup="{_esc(p.matchup)}" '
            f'data-seg="{_esc(p.segment)}" data-mkt="{_esc(p.market_type)}" '
            f'data-edge="{p.edge:.1f}" data-time="{_esc(p.time_cst)}" '
            f'data-rating="{p.confidence}">'
            f"<td {td}>{_esc(p.time_cst)}</td>"
            f"<td {td}>{matchup_cell}</td>"
            f"<td {td}>{_segment_pill(p.segment)}</td>"
            f'<td {td}><span style="font-weight:600;color:#374151;font-size:12px">{_esc(p.market_type)}</span></td>'
            f'<td style="padding:6px;border-bottom:1px solid #e9ecef;font-size:13px;'
            f'vertical-align:middle;border-left:3px solid {border_color};font-weight:700">'
            f"{_esc(p.label)}{odds_html}"
            f' <span style="font-size:11px;color:#9ca3af;font-weight:400">{_esc(p.market_line)}</span>'
            f"{rationale_html}</td>"
            f'<td {td}><span style="font-weight:600">{_esc(p.model_scores)}</span></td>'
            f'<td style="padding:6px;border-bottom:1px solid #e9ecef;text-align:center;'
            f'font-weight:700;font-size:14px;color:{edge_color}">{p.edge:.1f}</td>'
            f"<td {td}>{_confidence_badge(p.confidence)}</td>"
            "</tr>"
        )

    sort_arrow = (
        '<span class="sort-arrow" style="margin-left:4px;font-size:10px">\u25b2\u25bc</span>'
    )
    table = (
        '<table id="slate" style="width:100%;border-collapse:collapse;border:1px solid #dee2e6">'
        "<thead><tr>"
        f'<th style="{th_style}" onclick="sortTable(0,\'text\')" data-col="0">Time{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(1,\'text\')" data-col="1">Matchup{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(2,\'text\')" data-col="2">Seg{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(3,\'text\')" data-col="3">Market{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(4,\'text\')" data-col="4">Pick{sort_arrow}</th>'
        f'<th style="{th_style}" onclick="sortTable(5,\'text\')" data-col="5">Model{sort_arrow}</th>'
        f'<th style="{th_style};text-align:center" onclick="sortTable(6,\'num\')" data-col="6">Edge{sort_arrow}</th>'
        f'<th style="{th_style};text-align:center" onclick="sortTable(7,\'num\')" data-col="7">Rating{sort_arrow}</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )

    # ── Per-game odds source breakdown ────────────────────────
    odds_section = ""
    if odds_by_game:
        odds_rows: list[str] = []
        oth = (
            "background:#1a2332;color:#d4af37;font-weight:600;font-size:11px;"
            "letter-spacing:1px;text-transform:uppercase;padding:6px 8px;"
            "text-align:left;white-space:nowrap"
        )
        for gid, detail in odds_by_game.items():
            label = game_labels.get(gid, "")
            books = detail.get("books", {})
            if not books:
                continue
            first = True
            for bk, lines in sorted(books.items()):
                pieces: list[str] = []
                if "spread" in lines:
                    price = f" ({lines['spread_price']:+d})" if lines.get("spread_price") else ""
                    pieces.append(f"{lines['spread']:+.1f}{price}")
                else:
                    pieces.append("—")
                if "total" in lines:
                    price = f" ({lines['total_price']:+d})" if lines.get("total_price") else ""
                    pieces.append(f"{lines['total']:.1f}{price}")
                else:
                    pieces.append("—")
                if "home_ml" in lines:
                    pieces.append(f"{lines['home_ml']:+d}")
                else:
                    pieces.append("—")
                if "away_ml" in lines:
                    pieces.append(f"{lines['away_ml']:+d}")
                else:
                    pieces.append("—")
                # 1H markets
                if "spread_h1" in lines:
                    price = (
                        f" ({lines['spread_h1_price']:+d})" if lines.get("spread_h1_price") else ""
                    )
                    pieces.append(f"{lines['spread_h1']:+.1f}{price}")
                else:
                    pieces.append("—")
                if "total_h1" in lines:
                    price = (
                        f" ({lines['total_h1_price']:+d})" if lines.get("total_h1_price") else ""
                    )
                    pieces.append(f"{lines['total_h1']:.1f}{price}")
                else:
                    pieces.append("—")

                otd = 'style="padding:4px 8px;border-bottom:1px solid #e9ecef;font-size:12px"'
                matchup_cell = (
                    f"<td {otd}><b>{_esc(label)}</b></td>" if first else f"<td {otd}></td>"
                )
                odds_rows.append(
                    f"<tr>"
                    f"{matchup_cell}"
                    f"<td {otd}><b>{_esc(bk)}</b></td>"
                    f"<td {otd}>{_esc(pieces[0])}</td>"
                    f"<td {otd}>{_esc(pieces[1])}</td>"
                    f"<td {otd}>{_esc(pieces[2])}</td>"
                    f"<td {otd}>{_esc(pieces[3])}</td>"
                    f"<td {otd}>{_esc(pieces[4])}</td>"
                    f"<td {otd}>{_esc(pieces[5])}</td>"
                    f"</tr>"
                )
                first = False

        if odds_rows:
            # Timestamp from first game's detail
            any_detail = next(iter(odds_by_game.values()), {})
            ts_raw = any_detail.get("captured_at", "")
            ts_display = ""
            if ts_raw:
                try:
                    dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(_CST)
                    ts_display = f" &middot; As of {dt.strftime('%I:%M %p').lstrip('0')} CT"
                except Exception:
                    pass

            odds_section = (
                '<div style="margin-top:16px">'
                f'<div style="font-size:14px;font-weight:700;color:#1a2332;margin-bottom:4px">'
                f"\U0001f4ca Odds Sources{ts_display}</div>"
                '<table style="width:100%;border-collapse:collapse;border:1px solid #dee2e6">'
                f"<thead><tr>"
                f'<th style="{oth}">Game</th>'
                f'<th style="{oth}">Book</th>'
                f'<th style="{oth}">Spread</th>'
                f'<th style="{oth}">Total</th>'
                f'<th style="{oth}">Home ML</th>'
                f'<th style="{oth}">Away ML</th>'
                f'<th style="{oth}">1H Spread</th>'
                f'<th style="{oth}">1H Total</th>'
                f"</tr></thead>"
                f"<tbody>{''.join(odds_rows)}</tbody>"
                "</table></div>"
            )

    footer = (
        f'<div style="text-align:center;padding:8px;color:#9ca3af;font-size:11px">'
        f"GBSV NBA {_esc(MODEL_VERSION)} &middot; Sorted by edge descending</div>"
    )

    script = (
        "<script>"
        "var _sortCol=-1,_sortAsc=true;"
        "function sortTable(col,type){"
        "var tb=document.querySelector('#slate tbody');"
        "var rows=Array.from(tb.rows);"
        "if(_sortCol===col){_sortAsc=!_sortAsc}else{_sortCol=col;_sortAsc=type==='num'?false:true}"
        "rows.sort(function(a,b){"
        "var av=a.cells[col].textContent.trim(),bv=b.cells[col].textContent.trim();"
        "if(type==='num'){av=parseFloat(av)||0;bv=parseFloat(bv)||0;return _sortAsc?av-bv:bv-av}"
        "return _sortAsc?av.localeCompare(bv):bv.localeCompare(av);"
        "});"
        "rows.forEach(function(r,i){tb.appendChild(r);r.style.background=i%2===0?'#ffffff':'#f8f9fa'});"
        "document.querySelectorAll('#slate thead th .sort-arrow').forEach(function(s,i){"
        "s.textContent=i===col?(_sortAsc?'\\u25B2':'\\u25BC'):'\\u25B2\\u25BC';"
        "});"
        "}"
        "function applyFilters(){"
        "var m=document.getElementById('fMatchup').value,"
        "s=document.getElementById('fSeg').value,"
        "k=document.getElementById('fMkt').value,"
        "e=parseFloat(document.getElementById('fEdge').value)||0;"
        "var rows=document.querySelectorAll('#slate tbody tr');"
        "var vis=0;"
        "rows.forEach(function(r){"
        "var show=true;"
        "if(m&&r.dataset.matchup!==m)show=false;"
        "if(s&&r.dataset.seg!==s)show=false;"
        "if(k&&r.dataset.mkt!==k)show=false;"
        "if(e&&parseFloat(r.dataset.edge)<e)show=false;"
        "r.style.display=show?'':'none';"
        "if(show){r.style.background=vis%2===0?'#ffffff':'#f8f9fa';vis++;}"
        "});"
        "}"
        "function resetFilters(){"
        "document.getElementById('fMatchup').value='';"
        "document.getElementById('fSeg').value='';"
        "document.getElementById('fMkt').value='';"
        "document.getElementById('fEdge').value='';"
        "applyFilters();"
        "}"
        "</script>"
    )

    return header + filter_bar + table + odds_section + footer + script


# ── Legacy plain-text builder (kept for backward compat / tests) ──


def _format_game_line(pred: Any, game: Any) -> str:
    home_name = game.home_team.name if game.home_team is not None else f"Team {game.home_team_id}"
    away_name = game.away_team.name if game.away_team is not None else f"Team {game.away_team_id}"
    kickoff = game.commence_time
    kickoff_text = kickoff.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC") if kickoff else "TBD"

    fg_prob = float(pred.fg_home_ml_prob or 0.5)
    away_prob = 1.0 - fg_prob
    return (
        f"- {away_name} at {home_name} ({kickoff_text})\n"
        f"  FG spread {float(pred.fg_spread or 0.0):+.1f} | "
        f"FG total {float(pred.fg_total or 0.0):.1f} | "
        f"FG ML H {fg_prob:.3f} / A {away_prob:.3f}"
    )


def build_teams_text(predictions_with_games: list[tuple[Any, Any]], max_games: int) -> str:
    lines = ["NBA Predictions Update", ""]
    lines.append(f"Games: {min(len(predictions_with_games), max_games)}")
    lines.append("")

    for pred, game in predictions_with_games[:max_games]:
        lines.append(_format_game_line(pred, game))

    return "\n".join(lines)


# ── Senders ────────────────────────────────────────────────────────


async def send_alert(title: str, message: str, severity: str = "warning") -> None:
    """Send a lightweight alert to Teams via the configured webhook.

    Designed for critical system alerts (retrain failure, quota exhaustion,
    etc.).  Silently swallows delivery errors to avoid masking the original
    failure that triggered the alert.
    """
    from src.config import get_settings

    settings = get_settings()
    webhook_url = settings.teams_webhook_url
    if not webhook_url:
        logger.debug("No Teams webhook configured; skipping alert")
        return

    color = {"error": "attention", "warning": "warning"}.get(severity, "default")
    card: dict[str, Any] = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"⚠ {title}",
                            "weight": "bolder",
                            "size": "medium",
                            "color": color,
                        },
                        {
                            "type": "TextBlock",
                            "text": message,
                            "wrap": True,
                        },
                        {
                            "type": "TextBlock",
                            "text": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                            "size": "small",
                            "isSubtle": True,
                        },
                    ],
                },
            }
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(webhook_url, json=card)
            resp.raise_for_status()
        logger.info("Alert sent to Teams: %s", title)
    except Exception:
        logger.warning("Failed to send Teams alert: %s", title, exc_info=True)


def _payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False))


def _chunk_card_payload(
    payload: dict[str, Any],
    max_payload_bytes: int = 26_000,
) -> list[dict[str, Any]]:
    import copy

    if _payload_size_bytes(payload) <= max_payload_bytes:
        return [payload]

    try:
        original_content = payload["attachments"][0]["content"]
        original_body = list(original_content["body"])
    except (KeyError, IndexError, TypeError):
        return [payload]

    if len(original_body) <= 4:
        return [payload]

    shared_prefix = original_body[:4]
    variable_items = original_body[4:]
    chunks: list[dict[str, Any]] = []

    def _new_chunk() -> tuple[dict[str, Any], dict[str, Any]]:
        chunk_payload = copy.deepcopy(payload)
        chunk_content = chunk_payload["attachments"][0]["content"]
        chunk_content["body"] = list(shared_prefix)
        return chunk_payload, chunk_content

    chunk_payload, chunk_content = _new_chunk()
    variable_items_in_chunk = 0

    for item in variable_items:
        chunk_content["body"].append(item)
        if _payload_size_bytes(chunk_payload) > max_payload_bytes and variable_items_in_chunk > 0:
            chunk_content["body"].pop()
            chunk_content.pop("actions", None)
            chunks.append(chunk_payload)
            chunk_payload, chunk_content = _new_chunk()
            chunk_content["body"].append(item)
            variable_items_in_chunk = 1
            continue

        variable_items_in_chunk += 1

    chunks.append(chunk_payload)
    for chunk in chunks[:-1]:
        chunk["attachments"][0]["content"].pop("actions", None)
    return chunks


async def send_card_to_teams(webhook_url: str, payload: dict[str, Any]) -> None:
    """Send an Adaptive Card payload to a Teams incoming webhook.
    Automatically chunks the payload into multiple cards if it exceeds Power Automate's 28KB limit.
    """
    raw_size = _payload_size_bytes(payload)
    chunks = _chunk_card_payload(payload)
    if len(chunks) == 1:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
        return

    logger.info(
        "Payload is %d bytes (exceeds 25KB limit). Chunking Adaptive Card into %d parts...",
        raw_size,
        len(chunks),
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, chunk_payload in enumerate(chunks, start=1):
            response = await client.post(webhook_url, json=chunk_payload)
            response.raise_for_status()
            logger.info("Successfully sent Teams payload chunk %d/%d", i, len(chunks))


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

    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"

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


async def send_html_via_graph(team_id: str, channel_id: str, html_content: str) -> dict[str, Any]:
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

    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"

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
            f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/filesFolder"
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
