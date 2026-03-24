from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_predictor
from src.db.session import get_db
from src.db.repositories.models import ModelRepository
from src.models.predictor import Predictor
from src.models.versioning import MODEL_VERSION
from src.services.model import ModelService

router = APIRouter(prefix="/model", tags=["model"])

async def get_model_service(db: AsyncSession = Depends(get_db)):
    return ModelService(ModelRepository(db))

@router.get("/status")
async def model_status(predictor: Predictor = Depends(get_predictor)):
    runtime_status = predictor.get_runtime_status()
    return {
        "ready": predictor.is_ready,
        "version": MODEL_VERSION,
        "active_model_version": predictor.model_version,
        "runtime_status": runtime_status,
        "metrics": predictor.get_metrics(),
        "feature_importance": predictor.get_feature_importance(),
    }

@router.get("/registry")
async def model_registry(db: AsyncSession = Depends(get_db)):
    repo = ModelRepository(db)
    rows = await repo.get_all_models_ordered_by_creation()
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
    service: ModelService = Depends(get_model_service),
):
    return await service.get_performance(limit)

@router.post("/retrain")
async def retrain():
    from src.db.session import async_session_factory
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    async with async_session_factory() as db:
        metrics = await trainer.train(db)
    return {"status": "complete", "metrics": metrics}
