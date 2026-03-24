from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game, ModelRegistry, Prediction


class ModelRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_all_models_ordered_by_creation(self) -> Sequence[ModelRegistry]:
        result = await self._session.execute(select(ModelRegistry).order_by(ModelRegistry.created_at.desc()))
        return result.scalars().all()

    async def get_model_by_version(self, version: str) -> ModelRegistry | None:
        result = await self._session.execute(select(ModelRegistry).where(ModelRegistry.model_version == version))
        return result.scalar_one_or_none()
        
    async def get_finished_game_predictions(self):
        result = await self._session.execute(
            select(Prediction, Game)
            .join(Game, Prediction.game_id == Game.id)
            .where(
                Game.status == "FT",
                Game.home_score_fg.is_not(None),
                Game.away_score_fg.is_not(None),
            )
            .order_by(Game.commence_time.desc(), Prediction.predicted_at.desc())
        )
        return result.all()

