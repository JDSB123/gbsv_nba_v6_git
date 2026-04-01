"""Performance tracking – backtest edge profitability & prediction accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import Game, Prediction
from src.db.session import get_db
from src.services.prediction_integrity import (
    prediction_has_valid_score_payload,
    prediction_score_rank,
)

router = APIRouter(prefix="/performance", tags=["performance"])

# Must match teams.py constants
MIN_EDGE = 2.0
EDGE_THRESHOLDS = [2.0, 3.5, 5.0, 7.0, 9.0]
JUICE = 110  # standard -110 vig


def _get_nba_avg_total() -> float:
    try:
        from src.config import get_settings
        return get_settings().nba_avg_total
    except Exception:
        return 230.0


def _american_to_prob(odds_val: Any) -> float | None:
    """Convert American odds (int/float/str) to implied probability (0-1)."""
    if odds_val is None:
        return None
    try:
        odds = float(str(odds_val).replace("+", ""))
    except (ValueError, TypeError):
        return None
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def _consensus_ml_odds(books: dict, key: str) -> float | None:
    """Average a ML odds field across all books, return as float."""
    vals = [b[key] for b in books.values() if key in b and b[key] is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals))


# ── Graded pick ──────────────────────────────────────────────


@dataclass
class GradedPick:
    segment: str  # "FG" or "1H"
    market: str  # "SPREAD", "TOTAL", "ML"
    edge: float
    result: str  # "W", "L", "P"
    matchup: str = ""
    label: str = ""


# ── Grading helpers ──────────────────────────────────────────


def _grade_spread_ats(
    edge: float,
    actual_home: int,
    actual_away: int,
    market_spread: float,
) -> str:
    """Grade a spread pick ATS. edge > 0 means bet home to cover."""
    actual_margin = actual_home - actual_away
    ats_result = actual_margin + market_spread
    if ats_result == 0:
        return "P"
    if edge > 0:
        return "W" if ats_result > 0 else "L"
    return "W" if ats_result < 0 else "L"


def _grade_total(
    predicted_total: float,
    line: float,
    actual_home: int,
    actual_away: int,
) -> str:
    """Grade an over/under pick."""
    actual_total = actual_home + actual_away
    over = predicted_total > line
    if actual_total == line:
        return "P"
    if over:
        return "W" if actual_total > line else "L"
    return "W" if actual_total < line else "L"


def _grade_ml(fg_spread: float, actual_home: int, actual_away: int) -> str:
    """Grade a moneyline pick."""
    if actual_home == actual_away:
        return "P"
    if fg_spread > 0:
        return "W" if actual_home > actual_away else "L"
    return "W" if actual_away > actual_home else "L"


def _grade_1h_winner(h1_spread: float, actual_home_1h: int, actual_away_1h: int) -> str:
    """Grade a 1H spread pick (no market 1H line – graded as 1H winner)."""
    if actual_home_1h == actual_away_1h:
        return "P"
    if h1_spread > 0:
        return "W" if actual_home_1h > actual_away_1h else "L"
    return "W" if actual_away_1h > actual_home_1h else "L"


# ── Extract & grade all picks for a finished game ────────────


def _grade_game(pred: Any, game: Any) -> list[GradedPick]:
    """Produce graded picks for a single finished (pred, game) pair."""
    picks: list[GradedPick] = []

    fg_spread = float(pred.fg_spread or 0)
    fg_total = float(pred.fg_total or 0)
    h1_spread = float(getattr(pred, "h1_spread", 0) or 0)
    h1_total = float(getattr(pred, "h1_total", 0) or 0)

    actual_home_fg = int(cast(Any, game.home_score_fg))
    actual_away_fg = int(cast(Any, game.away_score_fg))
    actual_home_1h = int(cast(Any, game.home_score_1h)) if game.home_score_1h is not None else None
    actual_away_1h = int(cast(Any, game.away_score_1h)) if game.away_score_1h is not None else None

    home = game.home_team.name if game.home_team else f"Team {game.home_team_id}"
    away = game.away_team.name if game.away_team else f"Team {game.away_team_id}"
    matchup = f"{away} @ {home}"

    opening_spread = float(pred.opening_spread) if pred.opening_spread is not None else None
    opening_total = float(pred.opening_total) if pred.opening_total is not None else None

    # Extract 1H market lines from odds_sourced JSON
    odds_sourced = getattr(pred, "odds_sourced", None)
    if isinstance(odds_sourced, dict):
        opening_h1_spread_raw = odds_sourced.get("opening_h1_spread")
        opening_h1_total_raw = odds_sourced.get("opening_h1_total")
    else:
        opening_h1_spread_raw = None
        opening_h1_total_raw = None
    opening_h1_spread = (
        float(opening_h1_spread_raw)
        if opening_h1_spread_raw is not None
        else None
    )
    opening_h1_total = (
        float(opening_h1_total_raw)
        if opening_h1_total_raw is not None
        else None
    )

    nba_avg_total = _get_nba_avg_total()

    # ── FG Spread ATS ─────────────────────────────────────
    if opening_spread is not None and abs(opening_spread) >= 0.5:
        home_edge = opening_spread + fg_spread
        if abs(home_edge) >= MIN_EDGE:
            result = _grade_spread_ats(home_edge, actual_home_fg, actual_away_fg, opening_spread)
            label = f"{home} ATS" if home_edge > 0 else f"{away} ATS"
            picks.append(
                GradedPick("FG", "SPREAD", round(abs(home_edge), 1), result, matchup, label)
            )

    # ── FG Total O/U ──────────────────────────────────────
    if opening_total is not None:
        total_edge = abs(fg_total - opening_total)
        if total_edge >= MIN_EDGE:
            direction = "OVER" if fg_total > opening_total else "UNDER"
            result = _grade_total(fg_total, opening_total, actual_home_fg, actual_away_fg)
            picks.append(
                GradedPick(
                    "FG",
                    "TOTAL",
                    round(total_edge, 1),
                    result,
                    matchup,
                    f"{direction} {opening_total:.1f}",
                )
            )
    elif abs(fg_total - nba_avg_total) >= 5:
        total_edge = abs(fg_total - nba_avg_total)
        direction = "OVER" if fg_total > nba_avg_total else "UNDER"
        result = _grade_total(fg_total, nba_avg_total, actual_home_fg, actual_away_fg)
        picks.append(
            GradedPick(
                "FG",
                "TOTAL",
                round(total_edge, 1),
                result,
                matchup,
                f"{direction} {nba_avg_total:.1f}",
            )
        )

    # ── FG ML ─────────────────────────────────────────────
    fg_home_ml_prob = float(getattr(pred, "fg_home_ml_prob", 0) or 0.5)
    pick_home = fg_spread > 0
    win_prob = fg_home_ml_prob if pick_home else 1 - fg_home_ml_prob

    # Get market ML odds from odds_sourced books
    books = odds_sourced.get("books", {}) if isinstance(odds_sourced, dict) else {}
    if pick_home:
        ml_odds_val = _consensus_ml_odds(books, "home_ml")
    else:
        ml_odds_val = _consensus_ml_odds(books, "away_ml")
    market_prob = _american_to_prob(ml_odds_val)
    prob_edge = (win_prob - market_prob) if market_prob is not None else (win_prob - 0.5)

    if prob_edge > 0.02:  # Min 2% edge (matches teams.py)
        ml_pts_edge = round(prob_edge * 33.3, 1)
        result = _grade_ml(fg_spread, actual_home_fg, actual_away_fg)
        side = home if fg_spread > 0 else away
        picks.append(GradedPick("FG", "ML", ml_pts_edge, result, matchup, f"{side} ML"))

    # ── 1H markets ────────────────────────────────────────
    if actual_home_1h is not None and actual_away_1h is not None:
        # ── 1H Spread ATS (when market line available) ────
        if opening_h1_spread is not None and abs(opening_h1_spread) >= 0.5:
            h1_home_edge = opening_h1_spread + h1_spread
            if abs(h1_home_edge) >= MIN_EDGE:
                result = _grade_spread_ats(
                    h1_home_edge, actual_home_1h, actual_away_1h, opening_h1_spread
                )
                side = home if h1_home_edge > 0 else away
                picks.append(
                    GradedPick(
                        "1H", "SPREAD", round(abs(h1_home_edge), 1), result, matchup,
                        f"{side} 1H ATS {opening_h1_spread:+.1f}",
                    )
                )
        elif abs(h1_spread) >= MIN_EDGE:
            # No 1H market line — grade as 1H winner
            result = _grade_1h_winner(h1_spread, actual_home_1h, actual_away_1h)
            side = home if h1_spread > 0 else away
            picks.append(
                GradedPick("1H", "SPREAD", round(abs(h1_spread), 1), result, matchup, f"{side} 1H")
            )

        # ── 1H Total O/U ─────────────────────────────────
        if opening_h1_total is not None:
            h1_total_edge = abs(h1_total - opening_h1_total)
            if h1_total_edge >= MIN_EDGE:
                direction = "OVER" if h1_total > opening_h1_total else "UNDER"
                result = _grade_total(h1_total, opening_h1_total, actual_home_1h, actual_away_1h)
                picks.append(
                    GradedPick(
                        "1H",
                        "TOTAL",
                        round(h1_total_edge, 1),
                        result,
                        matchup,
                        f"1H {direction} {opening_h1_total:.1f}",
                    )
                )
        else:
            half_avg = nba_avg_total / 2
            h1_total_edge = abs(h1_total - half_avg)
            if h1_total_edge >= 4:
                direction = "OVER" if h1_total > half_avg else "UNDER"
                result = _grade_total(h1_total, half_avg, actual_home_1h, actual_away_1h)
                picks.append(
                    GradedPick(
                        "1H",
                        "TOTAL",
                        round(h1_total_edge, 1),
                        result,
                        matchup,
                        f"1H {direction} {h1_total:.1f}",
                    )
                )

    return picks


# ── Aggregate stats ──────────────────────────────────────────


@dataclass
class _Record:
    wins: int = 0
    losses: int = 0
    pushes: int = 0

    def add(self, result: str) -> None:
        if result == "W":
            self.wins += 1
        elif result == "L":
            self.losses += 1
        else:
            self.pushes += 1

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct(self) -> float | None:
        return round(self.wins / self.total * 100, 1) if self.total else None

    @property
    def roi(self) -> float | None:
        if not self.total:
            return None
        profit = self.wins * (100 / JUICE * 100) - self.losses * 100
        wagered = self.total * 100
        return round(profit / wagered * 100, 1)

    def to_dict(self) -> dict:
        return {
            "record": f"{self.wins}-{self.losses}" + (f"-{self.pushes}" if self.pushes else ""),
            "win_pct": self.win_pct,
            "roi_pct": self.roi,
            "picks": self.wins + self.losses + self.pushes,
        }


def _build_stats(graded: list[GradedPick]) -> dict:
    """Build full performance stats from graded picks."""
    # Overall record
    overall = _Record()
    for p in graded:
        overall.add(p.result)

    # By market type
    by_market: dict[str, _Record] = {}
    for p in graded:
        key = f"{p.segment}_{p.market}"
        by_market.setdefault(key, _Record()).add(p.result)

    # By edge threshold
    by_threshold: dict[str, _Record] = {}
    for threshold in EDGE_THRESHOLDS:
        rec = _Record()
        for p in graded:
            if p.edge >= threshold:
                rec.add(p.result)
        by_threshold[f"{threshold}+"] = rec

    return {
        "overall": overall.to_dict(),
        "by_market": {k: v.to_dict() for k, v in sorted(by_market.items())},
        "by_edge_threshold": {k: v.to_dict() for k, v in by_threshold.items()},
    }


def _score_accuracy(rows: list[tuple[Any, Any]]) -> dict:
    """Compute prediction score accuracy metrics."""
    fg_errors: list[float] = []
    spread_errors: list[float] = []
    total_errors: list[float] = []
    h1_errors: list[float] = []

    for pred, game in rows:
        home_fg = float(pred.predicted_home_fg or 0)
        away_fg = float(pred.predicted_away_fg or 0)
        actual_home = int(cast(Any, game.home_score_fg))
        actual_away = int(cast(Any, game.away_score_fg))

        fg_errors.append(abs(home_fg - actual_home))
        fg_errors.append(abs(away_fg - actual_away))
        spread_errors.append(abs((home_fg - away_fg) - (actual_home - actual_away)))
        total_errors.append(abs((home_fg + away_fg) - (actual_home + actual_away)))

        if game.home_score_1h is not None and game.away_score_1h is not None:
            home_1h = float(pred.predicted_home_1h or 0)
            away_1h = float(pred.predicted_away_1h or 0)
            h1_errors.append(abs(home_1h - int(cast(Any, game.home_score_1h))))
            h1_errors.append(abs(away_1h - int(cast(Any, game.away_score_1h))))

    result: dict[str, float] = {}
    if fg_errors:
        result["score_mae"] = round(sum(fg_errors) / len(fg_errors), 2)
    if spread_errors:
        result["spread_mae"] = round(sum(spread_errors) / len(spread_errors), 2)
    if total_errors:
        result["total_mae"] = round(sum(total_errors) / len(total_errors), 2)
    if h1_errors:
        result["h1_score_mae"] = round(sum(h1_errors) / len(h1_errors), 2)
    return result


def _clv_summary(rows: list[tuple[Any, Any]]) -> dict:
    """Summarize closing-line value from predictions that have CLV filled."""
    clv_spreads: list[float] = []
    clv_totals: list[float] = []
    for pred, _ in rows:
        if pred.clv_spread is not None:
            clv_spreads.append(float(pred.clv_spread))
        if pred.clv_total is not None:
            clv_totals.append(float(pred.clv_total))

    result: dict[str, Any] = {}
    if clv_spreads:
        result["spread"] = {
            "mean": round(sum(clv_spreads) / len(clv_spreads), 2),
            "positive_pct": round(sum(1 for c in clv_spreads if c > 0) / len(clv_spreads) * 100, 1),
            "sample_size": len(clv_spreads),
        }
    if clv_totals:
        result["total"] = {
            "mean": round(sum(clv_totals) / len(clv_totals), 2),
            "positive_pct": round(sum(1 for c in clv_totals if c > 0) / len(clv_totals) * 100, 1),
            "sample_size": len(clv_totals),
        }
    return result


# ── Recent results detail ────────────────────────────────────


def _recent_results(graded: list[GradedPick], limit: int = 50) -> list[dict]:
    return [
        {
            "matchup": p.matchup,
            "label": p.label,
            "segment": p.segment,
            "market": p.market,
            "edge": p.edge,
            "result": p.result,
        }
        for p in graded[-limit:]
    ]


def _latest_valid_score_rows(rows: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    seen: dict[int, tuple[Any, Any]] = {}
    ordered_game_ids: list[int] = []
    for pred, game in rows:
        gid = int(cast(Any, game.id))
        existing = seen.get(gid)
        if existing is None:
            ordered_game_ids.append(gid)
            seen[gid] = (pred, game)
            continue
        if prediction_score_rank(pred) > prediction_score_rank(existing[0]):
            seen[gid] = (pred, game)
    return [
        seen[gid]
        for gid in ordered_game_ids
        if prediction_has_valid_score_payload(seen[gid][0])
    ]


# ── API endpoints ────────────────────────────────────────────


@router.get("")
async def get_performance(
    db: AsyncSession = Depends(get_db),
    model_version: str | None = Query(None, description="Filter by model version"),
):
    """Return performance metrics for all completed games with predictions."""
    stmt = (
        select(Prediction, Game)
        .join(Game, Prediction.game_id == Game.id)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
        )
        .where(Game.status.in_(["FT", "AOT"]))
        .where(Game.home_score_fg.isnot(None))
        .where(Game.away_score_fg.isnot(None))
        .order_by(Game.commence_time)
    )
    if model_version is not None:
        stmt = stmt.where(Prediction.model_version == model_version)
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return {
            "status": "accumulating",
            "message": "No completed games with predictions yet. Performance data will appear after games finish.",
            "games_graded": 0,
        }

    unique_rows = _latest_valid_score_rows(rows)

    # Grade all picks
    all_graded: list[GradedPick] = []
    for pred, game in unique_rows:
        all_graded.extend(_grade_game(pred, game))

    return {
        "status": "ok",
        "games_graded": len(unique_rows),
        "picks_graded": len(all_graded),
        "accuracy": _score_accuracy(unique_rows),
        "pick_performance": _build_stats(all_graded),
        "clv": _clv_summary(unique_rows),
        "recent": _recent_results(all_graded),
    }


@router.get("/dashboard")
async def performance_dashboard(
    db: AsyncSession = Depends(get_db),
    model_version: str | None = Query(None, description="Filter by model version"),
):
    """Interactive HTML dashboard showing backtest performance."""
    stmt = (
        select(Prediction, Game)
        .join(Game, Prediction.game_id == Game.id)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
        )
        .where(Game.status.in_(["FT", "AOT"]))
        .where(Game.home_score_fg.isnot(None))
        .where(Game.away_score_fg.isnot(None))
        .order_by(Game.commence_time)
    )
    if model_version is not None:
        stmt = stmt.where(Prediction.model_version == model_version)
    result = await db.execute(stmt)
    rows = result.all()

    unique_rows = _latest_valid_score_rows(rows)

    # Grade
    all_graded: list[GradedPick] = []
    for pred, game in unique_rows:
        all_graded.extend(_grade_game(pred, game))

    html = _build_dashboard_html(unique_rows, all_graded)
    return Response(content=html, media_type="text/html")


# ── HTML dashboard builder ───────────────────────────────────


def _build_dashboard_html(
    rows: list[tuple[Any, Any]],
    graded: list[GradedPick],
) -> str:
    n_games = len(rows)
    n_picks = len(graded)
    accuracy = _score_accuracy(rows)
    stats = _build_stats(graded)
    clv = _clv_summary(rows)

    # Helper for stat cards
    def _card(title: str, body: str) -> str:
        return (
            f'<div class="card">'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-body">{body}</div>'
            f"</div>"
        )

    # ── Overview cards ────────────────────────────────────
    ov = stats["overall"]
    overview_cards = (
        _card("Games Graded", f'<span class="big">{n_games}</span>')
        + _card("Total Picks", f'<span class="big">{n_picks}</span>')
        + _card("Overall Record", f'<span class="big">{ov["record"]}</span>')
        + _card(
            "Win Rate",
            f'<span class="big {_pct_class(ov["win_pct"])}">{ov["win_pct"]:.1f}%</span>'
            if ov["win_pct"] is not None
            else '<span class="big">—</span>',
        )
        + _card(
            "ROI (flat -110)",
            f'<span class="big {_roi_class(ov["roi_pct"])}">{ov["roi_pct"]:+.1f}%</span>'
            if ov["roi_pct"] is not None
            else '<span class="big">—</span>',
        )
    )

    # ── Accuracy cards ────────────────────────────────────
    acc_html = ""
    for key, label in [
        ("score_mae", "Score MAE"),
        ("spread_mae", "Spread MAE"),
        ("total_mae", "Total MAE"),
        ("h1_score_mae", "1H Score MAE"),
    ]:
        val = accuracy.get(key)
        acc_html += _card(
            label,
            f'<span class="big">{val:.1f}</span>'
            if val is not None
            else '<span class="big">—</span>',
        )

    # ── By market table ───────────────────────────────────
    market_rows_html = ""
    for mkt, rec in sorted(stats["by_market"].items()):
        mkt_label = mkt.replace("_", " ")
        wp = f"{rec['win_pct']:.1f}%" if rec["win_pct"] is not None else "—"
        roi = f"{rec['roi_pct']:+.1f}%" if rec["roi_pct"] is not None else "—"
        roi_cls = _roi_class(rec["roi_pct"])
        wp_cls = _pct_class(rec["win_pct"])
        market_rows_html += (
            f"<tr><td>{mkt_label}</td><td>{rec['record']}</td>"
            f'<td class="{wp_cls}">{wp}</td>'
            f'<td class="{roi_cls}">{roi}</td></tr>'
        )

    # ── By threshold table ────────────────────────────────
    threshold_rows_html = ""
    for thr, rec in stats["by_edge_threshold"].items():
        wp = f"{rec['win_pct']:.1f}%" if rec["win_pct"] is not None else "—"
        roi = f"{rec['roi_pct']:+.1f}%" if rec["roi_pct"] is not None else "—"
        roi_cls = _roi_class(rec["roi_pct"])
        wp_cls = _pct_class(rec["win_pct"])
        threshold_rows_html += (
            f"<tr><td>Edge {thr}</td><td>{rec['record']}</td>"
            f"<td>{rec['picks']}</td>"
            f'<td class="{wp_cls}">{wp}</td>'
            f'<td class="{roi_cls}">{roi}</td></tr>'
        )

    # ── CLV section ───────────────────────────────────────
    clv_html = ""
    if clv:
        for mkt, data in clv.items():
            cls = "positive" if data["mean"] > 0 else "negative" if data["mean"] < 0 else ""
            clv_html += _card(
                f"CLV {mkt.title()}",
                f'<span class="big {cls}">{data["mean"]:+.2f}</span>'
                f"<br><small>{data['positive_pct']:.0f}% positive ({data['sample_size']} games)</small>",
            )
    else:
        clv_html = '<p class="subtle">CLV data will appear after games finish and closing lines are captured.</p>'

    # ── Recent results table ──────────────────────────────
    recent = graded[-50:]
    recent_html = ""
    for p in reversed(recent):
        r_cls = "win" if p.result == "W" else "loss" if p.result == "L" else "push"
        recent_html += (
            f'<tr class="{r_cls}-row">'
            f"<td>{p.matchup}</td>"
            f"<td>{p.label}</td>"
            f"<td>{p.segment} {p.market}</td>"
            f"<td>{p.edge}</td>"
            f'<td class="{r_cls}">{p.result}</td>'
            f"</tr>"
        )

    # ── Empty state ───────────────────────────────────────
    if not graded:
        body = (
            '<div class="empty">'
            "<h2>Accumulating Data</h2>"
            "<p>Performance metrics will appear here once games with predictions "
            "are completed. Check back after tonight's games finish!</p>"
            "</div>"
        )
    else:
        body = f"""
<section>
  <h2>Overview</h2>
  <div class="cards">{overview_cards}</div>
</section>

<section>
  <h2>Prediction Accuracy</h2>
  <div class="cards">{acc_html}</div>
</section>

<section>
  <h2>Performance by Market</h2>
  <table>
    <thead><tr><th>Market</th><th>Record</th><th>Win %</th><th>ROI</th></tr></thead>
    <tbody>{market_rows_html}</tbody>
  </table>
</section>

<section>
  <h2>Performance by Edge Threshold</h2>
  <p class="subtle">Higher edge thresholds filter to only the strongest picks.</p>
  <table>
    <thead><tr><th>Threshold</th><th>Record</th><th>Picks</th><th>Win %</th><th>ROI</th></tr></thead>
    <tbody>{threshold_rows_html}</tbody>
  </table>
</section>

<section>
  <h2>Closing Line Value (CLV)</h2>
  <div class="cards">{clv_html}</div>
</section>

<section>
  <h2>Recent Results</h2>
  <table>
    <thead><tr><th>Matchup</th><th>Pick</th><th>Type</th><th>Edge</th><th>Result</th></tr></thead>
    <tbody>{recent_html}</tbody>
  </table>
</section>
"""

    return (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>GBSV Performance Dashboard</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "margin:0;padding:16px;background:#f5f5f5;color:#1a2332}"
        ".container{max-width:1100px;margin:0 auto}"
        "h1{color:#1a2332;border-bottom:3px solid #d4af37;padding-bottom:8px;margin-bottom:24px}"
        "h2{color:#1a2332;margin-top:32px;margin-bottom:12px}"
        "section{background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;"
        "box-shadow:0 2px 8px rgba(0,0,0,.06)}"
        ".cards{display:flex;gap:12px;flex-wrap:wrap}"
        ".card{background:#f8f9fa;border-radius:8px;padding:16px;min-width:140px;flex:1}"
        ".card-title{font-size:12px;text-transform:uppercase;color:#6c757d;margin-bottom:6px;letter-spacing:.5px}"
        ".card-body{font-size:14px}"
        ".big{font-size:28px;font-weight:700}"
        ".positive{color:#28a745}.negative{color:#dc3545}"
        "table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}"
        "th{text-align:left;padding:8px 12px;background:#1a2332;color:#d4af37;font-size:11px;"
        "text-transform:uppercase;letter-spacing:.5px}"
        "td{padding:8px 12px;border-bottom:1px solid #eee}"
        "tr:hover{background:#f8f9fa}"
        ".win{color:#28a745;font-weight:700}.loss{color:#dc3545;font-weight:700}"
        ".push{color:#6c757d;font-weight:700}"
        ".win-row td:last-child{background:#d4edda}.loss-row td:last-child{background:#f8d7da}"
        ".subtle{color:#6c757d;font-size:13px}"
        ".empty{text-align:center;padding:60px 20px;color:#6c757d}"
        "@media(max-width:768px){.cards{flex-direction:column}.big{font-size:22px}"
        "table{font-size:11px}th,td{padding:6px 8px}}"
        "</style></head><body>"
        '<div class="container">'
        "<h1>GBSV Performance Dashboard</h1>"
        f"{body}"
        f'<p class="subtle" style="text-align:center;margin-top:24px">'
        f"{n_games} games graded &middot; {n_picks} picks tracked</p>"
        "</div></body></html>"
    )


def _pct_class(pct: float | None) -> str:
    if pct is None:
        return ""
    return "positive" if pct >= 52.4 else "negative" if pct < 48 else ""


def _roi_class(roi: float | None) -> str:
    if roi is None:
        return ""
    return "positive" if roi > 0 else "negative" if roi < 0 else ""
