"""Elo ratings: build, cache, and reset."""

import logging
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game
from src.models.elo import EloSystem
from src.models.features._helpers import _as_float, _as_str

logger = logging.getLogger(__name__)

# ── Shared Elo instance (built once per trainer/predictor run) ──
_elo_system: EloSystem | None = None


async def build_elo_ratings(db: AsyncSession) -> EloSystem:
    """Build Elo ratings from all completed games (called once per run)."""
    global _elo_system  # noqa: PLW0603
    if _elo_system is not None:
        return _elo_system

    res_1 = await db.execute(
        select(Game)
        .where(
            Game.status == "FT",
            Game.home_score_fg.is_not(None),
            Game.away_score_fg.is_not(None),
        )
        .order_by(Game.commence_time)
    )
    games = res_1.scalars().all()
    elo = EloSystem()
    for g in games:
        elo.update(
            int(cast(Any, g.home_team_id)),
            int(cast(Any, g.away_team_id)),
            _as_float(cast(Any, g.home_score_fg)),
            _as_float(cast(Any, g.away_score_fg)),
            season=_as_str(cast(Any, g.season)),
        )
    _elo_system = elo
    logger.info("Built Elo ratings from %d games", len(games))
    return elo


def reset_elo_cache() -> None:
    """Clear cached Elo so it's rebuilt on next call (e.g. after retrain)."""
    global _elo_system  # noqa: PLW0603
    _elo_system = None
