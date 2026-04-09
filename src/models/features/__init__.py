"""Feature engineering package for NBA prediction models.

Re-exports the public API from _core so existing imports like
``from src.models.features import build_feature_vector`` continue to work.
"""

from src.models.features._core import (  # noqa: F401
    INJURY_WEIGHTS,
    SHARP_BOOKS,
    SQUARE_BOOKS,
    TEAM_TZ,
    NaN,
    _as_float,
    _as_str,
    _home_spreads,
    _injury_features,
    build_elo_ratings,
    build_feature_vector,
    get_feature_columns,
    reset_elo_cache,
)

__all__ = [
    "build_feature_vector",
    "get_feature_columns",
    "reset_elo_cache",
    "build_elo_ratings",
    "NaN",
    "_as_float",
    "_as_str",
    "_home_spreads",
    "_injury_features",
    "INJURY_WEIGHTS",
    "TEAM_TZ",
    "SHARP_BOOKS",
    "SQUARE_BOOKS",
]
