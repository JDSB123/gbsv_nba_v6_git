"""Feature engineering package for NBA prediction model.

Public API (all existing imports keep working):
    build_feature_vector, get_feature_columns,
    build_elo_ratings, reset_elo_cache,
    _as_float, _as_str, _home_spreads,
    INJURY_WEIGHTS, SHARP_BOOKS, SQUARE_BOOKS, TEAM_TZ
"""

import logging
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.seasons import season_for_date
from src.db.models import Game

# ── Re-export public names for backward-compatible imports ──────
from src.models.features._elo import build_elo_ratings, reset_elo_cache
from src.models.features._game import (
    _derived_features,
    _elo_h2h_referee_features,
    _interaction_features,
    _venue_and_streak_features,
)
from src.models.features._helpers import (
    INJURY_WEIGHTS,
    SHARP_BOOKS,
    SQUARE_BOOKS,
    TEAM_TZ,
    _as_float,
    _as_str,
    _home_spreads,
)
from src.models.features._market import _market_features, _prop_consensus_features
from src.models.features._player import _player_and_quarter_features
from src.models.features._team import (
    _injury_features,
    _recent_form,
    _schedule_features,
    _team_season_stats,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_feature_vector",
    "get_feature_columns",
    "build_elo_ratings",
    "reset_elo_cache",
    "_as_float",
    "_as_str",
    "_home_spreads",
    "INJURY_WEIGHTS",
    "SHARP_BOOKS",
    "SQUARE_BOOKS",
    "TEAM_TZ",
]


async def build_feature_vector(
    game: Game,
    db: AsyncSession,
    odds_snapshots: list | None = None,
) -> dict[str, float] | None:
    """Build a feature dict for a single game.

    Args:
        game: The Game ORM object.
        db: Async database session (used for stats, recent form, injuries).
        odds_snapshots: Pre-fetched odds data.  When provided these are used
            instead of querying the ``OddsSnapshot`` table — this is the path
            used at prediction time so the model always runs on *fresh* odds.
            When ``None`` (training), historical cached odds from the DB are
            used instead.
    """
    home_id = int(cast(Any, game.home_team_id))
    away_id = int(cast(Any, game.away_team_id))
    season = _as_str(
        cast(Any, game.season),
        season_for_date(cast(Any, game.commence_time)),
    )
    home_name = game.home_team.name if game.home_team is not None else ""

    features: dict[str, float] = {}
    features.update(await _team_season_stats(db, home_id, away_id, season))
    features.update(await _recent_form(db, home_id, away_id, game))
    features.update(await _schedule_features(db, home_id, away_id, game))
    features.update(await _injury_features(db, home_id, away_id))
    features.update(await _player_and_quarter_features(db, home_id, away_id, game))
    features.update(await _prop_consensus_features(db, game, odds_snapshots))
    features.update(await _venue_and_streak_features(db, home_id, away_id, game, features))
    features.update(await _elo_h2h_referee_features(db, game, home_id, away_id))
    features.update(_derived_features(features, game))
    features.update(await _market_features(db, game, odds_snapshots, home_name))
    features.update(_interaction_features(features, game))

    return features


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature names for the model."""
    cols = []
    for prefix in ["home", "away"]:
        cols.extend(
            [
                f"{prefix}_ppg",
                f"{prefix}_oppg",
                f"{prefix}_wins",
                f"{prefix}_losses",
                f"{prefix}_pace",
                f"{prefix}_off_rating",
                f"{prefix}_def_rating",
                f"{prefix}_win_pct",
            ]
        )
        for label in ["l5", "l10"]:
            cols.extend(
                [
                    f"{prefix}_{label}_pts_avg",
                    f"{prefix}_{label}_pts_allowed_avg",
                    f"{prefix}_{label}_1h_pts_avg",
                    f"{prefix}_{label}_1h_allowed_avg",
                ]
            )
        cols.extend(
            [
                f"{prefix}_rest_days",
                f"{prefix}_b2b",
                f"{prefix}_games_7d",
                f"{prefix}_injury_impact",
                f"{prefix}_injured_count",
            ]
        )
        # Player-stat aggregated features
        cols.extend(
            [
                f"{prefix}_player_pts_avg",
                f"{prefix}_player_ast_avg",
                f"{prefix}_player_reb_avg",
                f"{prefix}_player_fg_pct",
                f"{prefix}_player_3pt_pct",
                f"{prefix}_starter_pts_avg",
                f"{prefix}_bench_pts_avg",
                f"{prefix}_bench_ratio",
                f"{prefix}_min_std",
            ]
        )
        # Quarter scoring tendencies
        cols.extend(
            [
                f"{prefix}_q1_avg",
                f"{prefix}_q3_avg",
            ]
        )
    # ── New per-team features ───────────────────────────────────
    for prefix in ["home", "away"]:
        cols.extend(
            [
                f"{prefix}_venue_ppg",
                f"{prefix}_venue_oppg",
                f"{prefix}_win_streak",
                f"{prefix}_l5_record",
                f"{prefix}_l10_record",
            ]
        )
    # ── Game-level features ─────────────────────────────────────
    cols.extend(
        [
            "expected_pace",
            "pace_diff",
            "season_progress",
            "home_elo",
            "away_elo",
            "elo_diff",
            "h2h_win_pct",
            "h2h_avg_margin",
            "tz_diff",
            "home_adj_off",
            "home_adj_def",
            "rest_diff",
        ]
    )
    # ── Referee history ─────────────────────────────────────────
    cols.extend(
        [
            "ref_avg_pts",
            "ref_home_win_pct_bias",
            "ref_over_pct",
        ]
    )
    # ── Market signals ──────────────────────────────────────────
    cols.extend(
        [
            "mkt_spread_avg",
            "mkt_spread_std",
            "mkt_total_avg",
            "mkt_total_std",
            "mkt_home_ml_prob",
            "mkt_1h_spread_avg",
            "mkt_1h_total_avg",
            "mkt_1h_home_ml_prob",
            # Sharp vs. Square analysis
            "sharp_spread",
            "square_spread",
            "sharp_square_spread_diff",
            "sharp_total",
            "square_total",
            "sharp_square_total_diff",
            "sharp_ml_prob",
            "square_ml_prob",
            "sharp_square_ml_diff",
            # Line movement & RLM
            "spread_move",
            "total_move",
            "rlm_flag",
        ]
    )
    # ── Player prop consensus signals ───────────────────────────
    cols.extend(
        [
            "prop_pts_lines_count",
            "prop_pts_avg_line",
            "prop_ast_avg_line",
            "prop_reb_avg_line",
            "prop_threes_avg_line",
            "prop_blk_avg_line",
            "prop_stl_avg_line",
            "prop_tov_avg_line",
            "prop_pra_avg_line",
            "prop_dd_count",
            "prop_td_count",
            "prop_sharp_square_diff",
        ]
    )
    # ── Matchup & Situational Styles ────────────────────────────
    cols.extend(
        [
            "matchup_3pt_diff",
            "situational_urgency",
        ]
    )
    # ── Cross-team interaction features ─────────────────────────
    cols.extend(
        [
            "pace_x_3pt_diff",
            "elo_x_rest",
            "injury_diff",
            "venue_scoring_edge",
            "off_def_mismatch",
            "streak_diff",
        ]
    )
    return cols
