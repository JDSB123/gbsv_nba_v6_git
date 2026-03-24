from datetime import datetime
from typing import Any, Sequence, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import Game, OddsSnapshot, Prediction, Team


class PredictionRepository:
    """Repository for accessing prediction and related game/odds data."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_latest_predictions_for_upcoming_games(self) -> Sequence[Prediction]:
        """Fetch the single latest prediction for all strictly upcoming (NS) games."""
        result = await self._session.execute(
            select(Prediction)
            .join(Game)
            .where(Game.status == "NS")
            .order_by(Game.commence_time, Prediction.predicted_at.desc())
        )
        predictions = result.scalars().all()

        seen: set[int] = set()
        latest: list[Prediction] = []
        for pred in predictions:
            game_id = int(cast(Any, pred.game_id))
            if game_id not in seen:
                seen.add(game_id)
                latest.append(pred)
        return latest

    async def get_games_with_teams_and_stats(self, game_ids: list[int]) -> Sequence[Game]:
        """Fetch games by ID, eagerly loading home/away teams AND their season stats."""
        result = await self._session.execute(
            select(Game)
            .options(
                selectinload(Game.home_team).selectinload(Team.season_stats),
                selectinload(Game.away_team).selectinload(Team.season_stats),
            )
            .where(Game.id.in_(game_ids))
            .order_by(Game.commence_time)
        )
        return result.scalars().all()

    async def get_games_with_teams(self, game_ids: list[int]) -> Sequence[Game]:
        """Fetch games by ID, eagerly loading just the home/away teams (no deep stats)."""
        result = await self._session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.id.in_(game_ids))
            .order_by(Game.commence_time)
        )
        return result.scalars().all()

    async def get_game_with_teams(self, game_id: int) -> Game | None:
        """Fetch a single game by ID, eagerly loading home/away teams."""
        result = await self._session.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.id == game_id)
        )
        return result.scalar_one_or_none()

    async def get_latest_prediction_for_game(self, game_id: int) -> Prediction | None:
        """Fetch the most recent prediction for a given game ID."""
        result = await self._session.execute(
            select(Prediction)
            .where(Prediction.game_id == game_id)
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_recent_odds_snapshots(
        self, game_id: int, limit: int = 50
    ) -> Sequence[OddsSnapshot]:
        """Fetch the most recent odds snapshots for a given game ID."""
        result = await self._session.execute(
            select(OddsSnapshot)
            .where(OddsSnapshot.game_id == game_id)
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_latest_odds_pull_timestamp(self) -> datetime | None:
        """Get the absolute latest timestamp of any odds data pulled."""
        result = await self._session.execute(select(func.max(OddsSnapshot.captured_at)))
        return result.scalar_one_or_none()
