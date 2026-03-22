from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.models.predictor import Predictor

predictor = Predictor()


async def get_predictor() -> Predictor:
    return predictor
