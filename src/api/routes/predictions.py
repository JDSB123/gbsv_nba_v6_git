from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import get_predictor
from src.db.models import Game, OddsSnapshot, Prediction
from src.db.session import get_db
from src.models.predictor import Predictor

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _format_prediction(pred: Prediction, game: Game) -> dict:
    home_name = game.home_team.name if game.home_team else f"Team {game.home_team_id}"
    away_name = game.away_team.name if game.away_team else f"Team {game.away_team_id}"
    return {
        "game_id": game.id,
        "odds_api_id": game.odds_api_id,
        "commence_time": game.commence_time.isoformat() if game.commence_time else None,
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
                "home_prob": pred.fg_home_ml_prob,
                "away_prob": round(1 - (pred.fg_home_ml_prob or 0.5), 3),
            },
            "h1_spread": {"prediction": pred.h1_spread},
            "h1_total": {"prediction": pred.h1_total},
            "h1_moneyline": {
                "home_prob": pred.h1_home_ml_prob,
                "away_prob": round(1 - (pred.h1_home_ml_prob or 0.5), 3),
            },
        },
        "model_version": pred.model_version,
        "predicted_at": pred.predicted_at.isoformat() if pred.predicted_at else None,
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
        if pred.game_id not in seen:
            seen.add(pred.game_id)
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
            o.point for o in odds if o.market == "spreads" and o.point is not None
        ]
        totals = [o.point for o in odds if o.market == "totals" and o.point is not None]

        if spreads:
            mkt_spread = float(np.mean(spreads))
            result["markets"]["fg_spread"]["market_line"] = mkt_spread
            result["markets"]["fg_spread"]["edge"] = round(
                pred.fg_spread - mkt_spread, 1
            )
        if totals:
            mkt_total = float(np.mean(totals))
            result["markets"]["fg_total"]["market_line"] = mkt_total
            result["markets"]["fg_total"]["edge"] = round(pred.fg_total - mkt_total, 1)

    return result
