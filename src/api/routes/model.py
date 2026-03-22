from typing import Any

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_predictor
from src.db.models import Game, ModelRegistry, Prediction
from src.db.session import get_db
from src.models.predictor import Predictor
from src.models.versioning import MODEL_VERSION

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/status")
async def model_status(predictor: Predictor = Depends(get_predictor)):
    """Return model version, metrics, and feature importance."""
    return {
        "ready": predictor.is_ready,
        "version": MODEL_VERSION,
        "active_model_version": predictor.model_version,
        "metrics": predictor.get_metrics(),
        "feature_importance": predictor.get_feature_importance(),
    }


@router.get("/registry")
async def model_registry(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    )
    rows = result.scalars().all()
    return {
        "models": [
            {
                "model_version": row.model_version,
                "is_active": row.is_active,
                "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
                "retired_at": row.retired_at.isoformat() if row.retired_at else None,
                "promotion_reason": row.promotion_reason,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


@router.get("/performance")
async def model_performance(
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """Realized post-game performance from finished games with latest predictions."""
    result = await db.execute(
        select(Prediction, Game)
        .join(Game, Prediction.game_id == Game.id)
        .where(
            Game.status == "FT",
            Game.home_score_fg.is_not(None),
            Game.away_score_fg.is_not(None),
        )
        .order_by(Game.commence_time.desc(), Prediction.predicted_at.desc())
    )
    rows = result.all()

    latest_per_game_version: dict[tuple[int, str], tuple[Prediction, Game]] = {}
    for pred, game in rows:
        key = (int(pred.game_id), str(pred.model_version))
        if key not in latest_per_game_version:
            latest_per_game_version[key] = (pred, game)
        if len(latest_per_game_version) >= limit:
            break

    by_model: dict[str, dict[str, list[float] | int]] = {}
    for pred, game in latest_per_game_version.values():
        model_version = str(pred.model_version)
        slot = by_model.setdefault(
            model_version,
            {
                "home_fg_abs": [],
                "away_fg_abs": [],
                "home_1h_abs": [],
                "away_1h_abs": [],
                "clv_spread": [],
                "clv_total": [],
                "count": 0,
            },
        )
        slot["count"] = int(slot["count"]) + 1

        slot["home_fg_abs"].append(
            abs(float(pred.predicted_home_fg) - float(game.home_score_fg))
        )
        slot["away_fg_abs"].append(
            abs(float(pred.predicted_away_fg) - float(game.away_score_fg))
        )

        if game.home_score_1h is not None and game.away_score_1h is not None:
            slot["home_1h_abs"].append(
                abs(float(pred.predicted_home_1h) - float(game.home_score_1h))
            )
            slot["away_1h_abs"].append(
                abs(float(pred.predicted_away_1h) - float(game.away_score_1h))
            )

        if pred.clv_spread is not None:
            slot["clv_spread"].append(float(pred.clv_spread))
        if pred.clv_total is not None:
            slot["clv_total"].append(float(pred.clv_total))

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return round(float(np.mean(values)), 4)

    performance: list[dict[str, Any]] = []
    for version, vals in by_model.items():
        home_fg = vals["home_fg_abs"]
        away_fg = vals["away_fg_abs"]
        home_1h = vals["home_1h_abs"]
        away_1h = vals["away_1h_abs"]
        performance.append(
            {
                "model_version": version,
                "sample_size": vals["count"],
                "mae_home_fg": _avg(home_fg),
                "mae_away_fg": _avg(away_fg),
                "mae_home_1h": _avg(home_1h),
                "mae_away_1h": _avg(away_1h),
                "mae_fg_combined": _avg(home_fg + away_fg),
                "mae_1h_combined": _avg(home_1h + away_1h),
                "avg_clv_spread": _avg(vals["clv_spread"]),
                "avg_clv_total": _avg(vals["clv_total"]),
            }
        )

    performance.sort(
        key=lambda row: row.get("sample_size", 0),
        reverse=True,
    )
    return {"window": limit, "models": performance}


@router.post("/retrain")
async def retrain():
    """Trigger a manual model retrain."""
    from src.db.session import async_session_factory
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    async with async_session_factory() as db:
        metrics = await trainer.train(db)
    return {"status": "complete", "metrics": metrics}
