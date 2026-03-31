from functools import lru_cache

from src.models.predictor import Predictor


@lru_cache
def _get_predictor() -> Predictor:
    return Predictor()


async def get_predictor() -> Predictor:
    return _get_predictor()


def reload_predictor() -> None:
    """Clear the cached Predictor so the next request rebuilds it (e.g. after retrain)."""
    _get_predictor.cache_clear()
