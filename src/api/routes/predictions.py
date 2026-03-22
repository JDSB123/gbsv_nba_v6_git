from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import get_predictor
from src.config import get_settings
from src.db.models import Game, OddsSnapshot, Prediction
from src.db.session import get_db
from src.models.predictor import Predictor
from src.notifications.teams import build_teams_text, send_text_to_teams

router = APIRouter(prefix="/predictions", tags=["predictions"])
settings = get_settings()


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _format_prediction(pred: Prediction, game: Game) -> dict:
    home_name = (
        game.home_team.name
        if game.home_team is not None
        else f"Team {game.home_team_id}"
    )
    away_name = (
        game.away_team.name
        if game.away_team is not None
        else f"Team {game.away_team_id}"
    )
    fg_home_ml_prob = _as_float(pred.fg_home_ml_prob, 0.5)
    h1_home_ml_prob = _as_float(pred.h1_home_ml_prob, 0.5)
    return {
        "game_id": game.id,
        "odds_api_id": game.odds_api_id,
        "commence_time": (
            game.commence_time.isoformat() if game.commence_time is not None else None
        ),
        "away_team": away_name,
        "home_team": home_name,
        "predicted_scores": {
            "full_game": {
                "away": pred.predicted_away_fg,
                "home": pred.predicted_home_fg,
            },
            "first_half": {
                "away": pred.predicted_away_1h,
                "home": pred.predicted_home_1h,
            },
        },
        "markets": {
            "fg_spread": {"prediction": pred.fg_spread},
            "fg_total": {"prediction": pred.fg_total},
            "fg_moneyline": {
                "home_prob": fg_home_ml_prob,
                "away_prob": round(1 - fg_home_ml_prob, 3),
            },
            "h1_spread": {"prediction": pred.h1_spread},
            "h1_total": {"prediction": pred.h1_total},
            "h1_moneyline": {
                "home_prob": h1_home_ml_prob,
                "away_prob": round(1 - h1_home_ml_prob, 3),
            },
        },
        "model_version": pred.model_version,
        "predicted_at": (
            pred.predicted_at.isoformat() if pred.predicted_at is not None else None
        ),
        "clv": {
            "opening_spread": pred.opening_spread,
            "opening_total": pred.opening_total,
            "closing_spread": pred.closing_spread,
            "closing_total": pred.closing_total,
            "clv_spread": pred.clv_spread,
            "clv_total": pred.clv_total,
        },
    }


@router.get("")
async def list_predictions(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Get latest predictions for all upcoming games."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Subquery: latest prediction per game
    result = await db.execute(
        select(Prediction)
        .join(Game)
        .where(Game.status == "NS")
        .order_by(Game.commence_time, Prediction.predicted_at.desc())
    )
    predictions = result.scalars().all()

    # Deduplicate to latest per game
    seen: set[int] = set()
    latest: list[Prediction] = []
    for pred in predictions:
        game_id = int(cast(Any, pred.game_id))
        if game_id not in seen:
            seen.add(game_id)
            latest.append(pred)

    output = []
    for pred in latest:
        game_result = await db.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.id == pred.game_id)
        )
        game = game_result.scalar_one_or_none()
        if game:
            output.append(_format_prediction(pred, game))

    return {"predictions": output, "count": len(output)}


@router.get("/{game_id}")
async def get_prediction(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Get prediction detail for a specific game."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    game_result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.id == game_id)
    )
    game = game_result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    pred_result = await db.execute(
        select(Prediction)
        .where(Prediction.game_id == game_id)
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
    )
    pred = pred_result.scalar_one_or_none()
    if not pred:
        raise HTTPException(
            status_code=404, detail="No prediction available for this game"
        )

    result = _format_prediction(pred, game)

    # Add latest odds for edge comparison
    odds_result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.game_id == game_id)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(50)
    )
    odds = odds_result.scalars().all()
    if odds:
        import numpy as np

        spreads = [
            float(cast(Any, o.point))
            for o in odds
            if cast(Any, o.market) == "spreads" and o.point is not None
        ]
        totals = [
            float(cast(Any, o.point))
            for o in odds
            if cast(Any, o.market) == "totals" and o.point is not None
        ]

        if spreads:
            mkt_spread = float(np.mean(spreads))
            result["markets"]["fg_spread"]["market_line"] = mkt_spread
            result["markets"]["fg_spread"]["edge"] = round(
                _as_float(pred.fg_spread) - mkt_spread, 1
            )
        if totals:
            mkt_total = float(np.mean(totals))
            result["markets"]["fg_total"]["market_line"] = mkt_total
            result["markets"]["fg_total"]["edge"] = round(
                _as_float(pred.fg_total) - mkt_total, 1
            )

    return result


@router.post("/publish/teams")
async def publish_predictions_to_teams(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Generate latest predictions and send formatted payload to Teams."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not settings.teams_webhook_url:
        raise HTTPException(status_code=400, detail="Teams webhook is not configured")

    predictions = await predictor.predict_upcoming(db)
    if not predictions:
        return {"status": "ok", "published": 0, "detail": "No upcoming games"}

    game_ids = [int(cast(Any, p.game_id)) for p in predictions]
    game_result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.id.in_(game_ids))
        .order_by(Game.commence_time)
    )
    games = game_result.scalars().all()
    game_by_id = {int(cast(Any, g.id)): g for g in games}

    rows: list[tuple[Prediction, Game]] = []
    for pred in predictions:
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game is not None:
            rows.append((pred, game))

    text = build_teams_text(rows, settings.teams_max_games_per_message)
    await send_text_to_teams(settings.teams_webhook_url, text)
    return {"status": "ok", "published": len(rows)}
