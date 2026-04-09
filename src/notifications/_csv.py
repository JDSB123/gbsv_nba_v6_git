"""CSV slate builder for NBA pick exports."""

from __future__ import annotations

import csv
import io
from typing import Any

from src.models.odds_utils import consensus_line as _consensus_line
from src.notifications._helpers import MIN_EDGE, Pick
from src.notifications._picks import extract_picks


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
        cons_spread = _consensus_line(books, "spread")
        cons_total = _consensus_line(books, "total")
        cons_parts = []
        if cons_spread is not None:
            cons_parts.append(f"S:{cons_spread:+.1f}")
        if cons_total is not None:
            cons_parts.append(f"T:{cons_total:.1f}")
        odds_source_str = (
            f"consensus({'/'.join(cons_parts)}) [{len(books)} books]" if cons_parts else ""
        )
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
