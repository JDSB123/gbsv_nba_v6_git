"""Pick extraction: derive actionable picks with edge values from predictions."""

from __future__ import annotations

from typing import Any

from src.models.odds_utils import (
    american_to_prob,
    consensus_line,
    consensus_price,
    prob_to_american,
)
from src.notifications._helpers import (
    MIN_EDGE,
    Pick,
    _fire_count,
    _fmt_time_cst,
    _get_nba_avg_total,
    _team_record,
)

# Keep private aliases so the body reads naturally
_consensus_line = consensus_line
_consensus_price = consensus_price
_prob_to_american = prob_to_american
_american_to_prob = american_to_prob


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
    mkt_spread = (
        float(opening_spread)
        if opening_spread is not None and abs(float(opening_spread)) >= 0.5
        else _consensus_line(books, "spread")
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
                    label, e, t_cst, matchup, "FG", "SPREAD", f"{mkt_spread:+.1f}",
                    fg_scores, h_rec, a_rec, _fire_count(e),
                    odds=odds_map.get("FG_SPREAD", ""), rationale=rationale,
                )
            )
    elif abs(fg_spread) >= min_edge:
        pick_home = fg_spread > 0
        label = f"{home} {-fg_spread:+.1f}" if pick_home else f"{away} {fg_spread:+.1f}"
        e = round(abs(fg_spread), 1)
        side_name = home if pick_home else away
        rationale = f"No market line · Model projects {side_name} by {e:.1f}pts"
        picks.append(
            Pick(
                label, e, t_cst, matchup, "FG", "SPREAD", f"{-fg_spread:+.1f}",
                fg_scores, h_rec, a_rec, _fire_count(e),
                odds=odds_map.get("FG_SPREAD", ""), rationale=rationale,
            )
        )

    # ── Full-game total ───────────────────────────────────────
    mkt_total = (
        float(opening_total)
        if opening_total is not None
        else _consensus_line(books, "total")
    )
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
                    f"{direction} {mkt_total:.1f}", e, t_cst, matchup, "FG", "TOTAL",
                    f"{mkt_total:.1f}", fg_scores, h_rec, a_rec, _fire_count(e),
                    odds=odds_map.get("FG_TOTAL", ""), rationale=rationale,
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
                    f"{direction} {fg_total:.1f}", e, t_cst, matchup, "FG", "TOTAL",
                    f"{fg_total:.1f}", fg_scores, h_rec, a_rec, _fire_count(e),
                    odds=odds_map.get("FG_TOTAL", ""), rationale=rationale,
                )
            )

    # ── 1H spread ─────────────────────────────────────────────
    mkt_h1_spread = (
        float(opening_h1_spread)
        if opening_h1_spread is not None and abs(float(opening_h1_spread)) >= 0.5
        else _consensus_line(books, "spread_h1")
    )
    if mkt_h1_spread is not None:
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
                    label, e, t_cst, matchup, "1H", "SPREAD", f"{mkt_h1_spread:+.1f}",
                    h1_scores, h_rec, a_rec, _fire_count(e),
                    odds=odds_map.get("1H_SPREAD", ""), rationale=rationale,
                )
            )
    elif abs(h1_spread) >= min_edge:
        pick_home = h1_spread > 0
        label = f"{home} {-h1_spread:+.1f}" if pick_home else f"{away} {h1_spread:+.1f}"
        e = round(abs(h1_spread), 1)
        side_name = home if pick_home else away
        rationale = f"No 1H line · Model projects {side_name} 1H by {e:.1f}pts"
        picks.append(
            Pick(
                label, e, t_cst, matchup, "1H", "SPREAD", f"{-h1_spread:+.1f}",
                h1_scores, h_rec, a_rec, _fire_count(e),
                odds=odds_map.get("1H_SPREAD", ""), rationale=rationale,
            )
        )

    # ── 1H total ──────────────────────────────────────────────
    mkt_h1_total = (
        float(opening_h1_total)
        if opening_h1_total is not None
        else _consensus_line(books, "total_h1")
    )
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
                    f"{direction} {mkt_h1_total:.1f}", e, t_cst, matchup, "1H", "TOTAL",
                    f"{mkt_h1_total:.1f}", h1_scores, h_rec, a_rec, _fire_count(e),
                    odds=odds_map.get("1H_TOTAL", ""), rationale=rationale,
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
                    f"{direction} {h1_total:.1f}", e, t_cst, matchup, "1H", "TOTAL",
                    f"{h1_total:.1f}", h1_scores, h_rec, a_rec, _fire_count(e),
                    odds=odds_map.get("1H_TOTAL", ""), rationale=rationale,
                )
            )

    # ── FG ML pick ─────────────────────────────────────────────
    pick_home = fg_spread > 0
    side = home if pick_home else away
    win_prob = fg_prob if pick_home else 1 - fg_prob

    if pick_home:
        ml_odds = odds_map.get("FG_ML_HOME", "") or _prob_to_american(fg_prob)
    else:
        ml_odds = odds_map.get("FG_ML_AWAY", "") or _prob_to_american(1 - fg_prob)

    market_prob = _american_to_prob(ml_odds)
    prob_edge = (win_prob - market_prob) if market_prob is not None else (win_prob - 0.5)

    if prob_edge > 0.02:
        ml_pts_edge = round(prob_edge * 33.3, 1)
        m_prob_str = f"{market_prob:.0%}" if market_prob is not None else "N/A"
        rationale = (
            f"Model projects {side} by {abs(fg_spread):.1f}pts. "
            f"Edge: {win_prob:.0%} win prob vs {m_prob_str} implied ({ml_odds or 'n/a'})"
        )
        picks.append(
            Pick(
                f"{side} ML", ml_pts_edge, t_cst, matchup, "FG", "ML",
                ml_odds or _prob_to_american(win_prob), fg_scores, h_rec, a_rec,
                _fire_count(ml_pts_edge), odds=ml_odds, rationale=rationale,
            )
        )

    # ── 1H ML pick ────────────────────────────────────────────
    h1_prob = float(getattr(pred, "h1_home_ml_prob", 0) or 0.5)
    h1_pick_home = h1_spread > 0
    h1_side = home if h1_pick_home else away
    h1_win_prob = h1_prob if h1_pick_home else 1 - h1_prob

    if h1_pick_home:
        h1_ml_odds = odds_map.get("1H_ML_HOME", "") or _prob_to_american(h1_prob)
    else:
        h1_ml_odds = odds_map.get("1H_ML_AWAY", "") or _prob_to_american(1 - h1_prob)

    h1_market_prob = _american_to_prob(h1_ml_odds)
    h1_prob_edge = (h1_win_prob - h1_market_prob) if h1_market_prob is not None else (h1_win_prob - 0.5)

    if h1_prob_edge > 0.02:
        h1_ml_pts_edge = round(h1_prob_edge * 33.3, 1)
        h1_m_prob_str = f"{h1_market_prob:.0%}" if h1_market_prob is not None else "N/A"
        h1_rationale = (
            f"1H Model projects {h1_side} by {abs(h1_spread):.1f}pts. "
            f"Edge: {h1_win_prob:.0%} win prob vs {h1_m_prob_str} implied ({h1_ml_odds or 'n/a'})"
        )
        picks.append(
            Pick(
                f"{h1_side} 1H ML", h1_ml_pts_edge, t_cst, matchup, "1H", "ML",
                h1_ml_odds or _prob_to_american(h1_win_prob), h1_scores, h_rec, a_rec,
                _fire_count(h1_ml_pts_edge), odds=h1_ml_odds, rationale=h1_rationale,
            )
        )

    return picks
