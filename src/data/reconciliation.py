"""Game reconciliation — merge synthetic (odds-only) games into official ones."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import case, delete, func, select, update

from src.db.models import (
    Game,
    GameReferee,
    OddsSnapshot,
    PlayerGameStats,
    Prediction,
    RotationChange,
)
from src.services.prediction_integrity import prediction_rank

logger = logging.getLogger(__name__)

_GAME_MATCH_WINDOW = timedelta(hours=12)


def _copy_prediction_payload(
    target: Prediction,
    preferred: Prediction,
    fallback: Prediction | None = None,
) -> None:
    fields = [
        "predicted_home_fg",
        "predicted_away_fg",
        "predicted_home_1h",
        "predicted_away_1h",
        "fg_spread",
        "fg_total",
        "fg_home_ml_prob",
        "h1_spread",
        "h1_total",
        "h1_home_ml_prob",
        "opening_spread",
        "opening_total",
        "closing_spread",
        "closing_total",
        "clv_spread",
        "clv_total",
        "odds_sourced",
        "predicted_at",
    ]
    for field in fields:
        preferred_value = getattr(preferred, field)
        if preferred_value is not None:
            setattr(target, field, preferred_value)
            continue
        if fallback is not None:
            setattr(target, field, getattr(fallback, field))


async def _find_matching_game(
    db: Any,
    home_team_id: int | None,
    away_team_id: int | None,
    commence_time: datetime,
    *,
    real_only: bool = False,
    exclude_game_id: int | None = None,
) -> Game | None:
    if home_team_id is None or away_team_id is None:
        return None

    filters = [
        Game.home_team_id == home_team_id,
        Game.away_team_id == away_team_id,
        Game.commence_time.between(
            commence_time - _GAME_MATCH_WINDOW,
            commence_time + _GAME_MATCH_WINDOW,
        ),
    ]
    if real_only:
        filters.append(Game.id > 0)
    if exclude_game_id is not None:
        filters.append(Game.id != exclude_game_id)

    result = await db.execute(
        select(Game)
        .where(*filters)
        .order_by(
            case((Game.id > 0, 0), else_=1),
            func.abs(func.extract("epoch", Game.commence_time - commence_time)),
            Game.commence_time,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _merge_game_records(db: Any, source_game: Game, target_game: Game) -> bool:
    if source_game.id == target_game.id:
        return False

    if source_game.id > 0 and target_game.id < 0:
        source_game, target_game = target_game, source_game

    source_id = int(cast(Any, source_game.id))
    target_id = int(cast(Any, target_game.id))
    if source_id > 0 or target_id < 0:
        logger.warning(
            "Refusing non-canonical game merge: source=%s target=%s",
            source_id,
            target_id,
        )
        return False

    source_odds_id = cast(Any, source_game.odds_api_id)
    target_odds_id = cast(Any, target_game.odds_api_id)
    if target_odds_id is None and source_odds_id is not None:
        source_game.odds_api_id = None
        target_game.odds_api_id = source_odds_id
    elif target_odds_id not in (None, source_odds_id) and source_odds_id is not None:
        logger.warning(
            "Conflicting odds_api_id during merge: keeping real game %s=%s and dropping synthetic %s=%s",
            target_id,
            target_odds_id,
            source_id,
            source_odds_id,
        )

    target_preds = (
        await db.execute(select(Prediction).where(Prediction.game_id == target_id))
    ).scalars().all()
    source_preds = (
        await db.execute(select(Prediction).where(Prediction.game_id == source_id))
    ).scalars().all()
    target_by_version = {
        str(cast(Any, pred.model_version)): pred for pred in target_preds if pred.model_version is not None
    }

    for source_pred in source_preds:
        model_version = str(cast(Any, source_pred.model_version))
        existing = target_by_version.get(model_version)
        if existing is None:
            source_pred.game_id = target_id
            continue

        preferred = max((existing, source_pred), key=prediction_rank)
        fallback = source_pred if preferred is existing else existing
        _copy_prediction_payload(existing, preferred, fallback)
        await db.delete(source_pred)

    target_player_ids = set(
        (
            await db.execute(
                select(PlayerGameStats.player_id).where(PlayerGameStats.game_id == target_id)
            )
        ).scalars().all()
    )
    if target_player_ids:
        await db.execute(
            delete(PlayerGameStats).where(
                PlayerGameStats.game_id == source_id,
                PlayerGameStats.player_id.in_(target_player_ids),
            )
        )

    await db.execute(
        update(OddsSnapshot)
        .where(OddsSnapshot.game_id == source_id)
        .values(game_id=target_id)
    )
    await db.execute(
        update(PlayerGameStats)
        .where(PlayerGameStats.game_id == source_id)
        .values(game_id=target_id)
    )
    await db.execute(
        update(GameReferee)
        .where(GameReferee.game_id == source_id)
        .values(game_id=target_id)
    )
    await db.execute(
        update(RotationChange)
        .where(RotationChange.game_id == source_id)
        .values(game_id=target_id)
    )

    await db.flush()
    await db.delete(source_game)
    logger.info(
        "Reconciled synthetic game %s into official game %s",
        source_id,
        target_id,
    )
    return True


async def reconcile_duplicate_games(db: Any, lookback_days: int | None = 7) -> int:
    query = select(Game).where(Game.id < 0).order_by(Game.commence_time)
    if lookback_days is not None:
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=lookback_days)
        query = query.where(Game.commence_time >= cutoff)

    synthetic_games = (await db.execute(query)).scalars().all()
    reconciled = 0
    for synthetic_game in synthetic_games:
        real_match = await _find_matching_game(
            db,
            int(cast(Any, synthetic_game.home_team_id)),
            int(cast(Any, synthetic_game.away_team_id)),
            cast(Any, synthetic_game.commence_time),
            real_only=True,
            exclude_game_id=int(cast(Any, synthetic_game.id)),
        )
        if real_match is None:
            continue
        if await _merge_game_records(db, synthetic_game, real_match):
            reconciled += 1
    return reconciled
