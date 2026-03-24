from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game, OddsSnapshot


class OddsRepository:
    # Repository for odds data access.

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_latest_odds_for_upcoming_games(self, limit: int = 500) -> Sequence[OddsSnapshot]:
        # Fetch the most recent odds snapshots for unstarted (NS) games.
        result = await self._session.execute(
            select(OddsSnapshot)
            .join(Game)
            .where(Game.status == "NS")
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
