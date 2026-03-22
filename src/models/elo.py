"""Rolling Elo power-rating system for NBA teams.

Computes Elo ratings from completed games using the standard Elo update
formula with home-court advantage and margin-of-victory adjustments.
Designed to be called once during dataset build (trainer) or feature
construction and reused across all games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 100.0  # ~100 Elo points ≈ ~3.5 point NBA home edge
SEASON_REVERSION = 0.75  # revert 25 % toward mean between seasons


@dataclass
class EloSystem:
    """Maintains rolling Elo ratings for every team."""

    ratings: dict[int, float] = field(default_factory=dict)
    _current_season: str | None = field(default=None, repr=False)

    def _get(self, team_id: int) -> float:
        return self.ratings.setdefault(team_id, INITIAL_ELO)

    def _mov_multiplier(self, margin: float) -> float:
        """Margin-of-victory multiplier (log-based, capped)."""
        return math.log(max(abs(margin), 1) + 1) * 0.8

    def _season_reset(self, season: str) -> None:
        """Revert ratings toward the mean at the start of a new season."""
        if self._current_season is not None and season != self._current_season:
            for tid in self.ratings:
                self.ratings[tid] = (
                    INITIAL_ELO + (self.ratings[tid] - INITIAL_ELO) * SEASON_REVERSION
                )
        self._current_season = season

    def update(
        self,
        home_id: int,
        away_id: int,
        home_score: float,
        away_score: float,
        season: str = "",
    ) -> tuple[float, float]:
        """Process one game result.  Returns (home_elo_before, away_elo_before)."""
        if season:
            self._season_reset(season)

        home_elo = self._get(home_id)
        away_elo = self._get(away_id)

        # Expected scores (home gets a boost)
        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE) / 400))
        exp_away = 1.0 - exp_home

        # Actual result
        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif away_score > home_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        margin = home_score - away_score
        mov = self._mov_multiplier(margin)

        self.ratings[home_id] = home_elo + K_FACTOR * mov * (actual_home - exp_home)
        self.ratings[away_id] = away_elo + K_FACTOR * mov * (actual_away - exp_away)

        return home_elo, away_elo

    def rating(self, team_id: int) -> float:
        return self._get(team_id)
