"""Plain-text slate builder (legacy, kept for backward compat / tests)."""

from __future__ import annotations

from datetime import UTC
from typing import Any


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
