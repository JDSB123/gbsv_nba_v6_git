"""Shared constants and utility functions for feature engineering."""

import logging
import math
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

NaN = float("nan")


def _as_float(value: Any, default: float = NaN) -> float:
    return float(value) if value is not None else default


def _as_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


def _assigned_referee_names(game: Any) -> list[str]:
    """Return assigned referee names when the relation is present."""
    referees = getattr(game, "referees", None) or []
    return [
        _as_str(getattr(referee, "referee_name", "")).strip()
        for referee in referees
        if _as_str(getattr(referee, "referee_name", "")).strip()
    ]


def _home_spreads(
    snapshots: Sequence[Any],
    home_team_name: str,
    market: str = "spreads",
    books: frozenset[str] | None = None,
) -> list[float]:
    """Extract the home team's spread values from odds snapshots.

    The Odds API returns two outcomes per bookmaker (one per team) with
    opposite-signed point values.  Averaging both cancels to ~0 and
    discards all directional information.

    This function keeps only the home team's outcome, preserving the
    standard betting convention::

        negative  -> home favorite  (e.g. -5.5 means home gives 5.5)
        positive  -> home underdog  (e.g. +3.0 means home gets 3)

    Parameters
    ----------
    books : optional frozenset — only include bookmakers in this set.
    """
    result: list[float] = []
    for s in snapshots:
        s_market = getattr(s, "market", None)
        s_outcome = getattr(s, "outcome_name", None)
        if s_market is None or s_outcome is None:
            continue
        if _as_str(s_market) != market or s.point is None:
            continue
        if _as_str(s_outcome) != home_team_name:
            continue
        if books is not None and _as_str(s.bookmaker).lower() not in books:
            continue
        result.append(_as_float(s.point))
    return result


# Injury status weights
INJURY_WEIGHTS = {"out": 1.0, "doubtful": 0.75, "questionable": 0.25, "probable": 0.05}

# ── NBA team timezone offsets (UTC) for travel features ─────
TEAM_TZ: dict[str, int] = {
    # Eastern (-5)
    "Boston Celtics": -5,
    "Brooklyn Nets": -5,
    "New York Knicks": -5,
    "Philadelphia 76ers": -5,
    "Toronto Raptors": -5,
    "Chicago Bulls": -6,
    "Cleveland Cavaliers": -5,
    "Detroit Pistons": -5,
    "Indiana Pacers": -5,
    "Milwaukee Bucks": -6,
    "Atlanta Hawks": -5,
    "Charlotte Hornets": -5,
    "Miami Heat": -5,
    "Orlando Magic": -5,
    "Washington Wizards": -5,
    # Central (-6)
    "Dallas Mavericks": -6,
    "Houston Rockets": -6,
    "Memphis Grizzlies": -6,
    "New Orleans Pelicans": -6,
    "San Antonio Spurs": -6,
    "Minnesota Timberwolves": -6,
    "Oklahoma City Thunder": -6,
    # Mountain (-7)
    "Denver Nuggets": -7,
    "Utah Jazz": -7,
    "Phoenix Suns": -7,
    # Pacific (-8)
    "Golden State Warriors": -8,
    "LA Clippers": -8,
    "Los Angeles Lakers": -8,
    "Portland Trail Blazers": -8,
    "Sacramento Kings": -8,
}

# ── Sharp vs. Square book classification ────────────────────
SHARP_BOOKS = frozenset(
    {
        "pinnacle",
        "lowvig",
        "betonlineag",
    }
)
SQUARE_BOOKS = frozenset(
    {
        "fanduel",
        "draftkings",
        "betmgm",
        "pointsbetus",
        "caesars",
        "wynnbet",
        "unibet_us",
        "betrivers",
        "superbook",
        "twinspires",
        "betus",
    }
)
