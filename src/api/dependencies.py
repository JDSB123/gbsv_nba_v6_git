from functools import lru_cache

from src.models.predictor import Predictor


@lru_cache
def _get_predictor() -> Predictor:
    return Predictor()


async def get_predictor() -> Predictor:
    return _get_predictor()
