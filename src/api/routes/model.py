from fastapi import APIRouter, Depends

from src.api.dependencies import get_predictor
from src.models.predictor import MODEL_VERSION, Predictor

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/status")
async def model_status(predictor: Predictor = Depends(get_predictor)):
    """Return model version, metrics, and feature importance."""
    return {
        "ready": predictor.is_ready,
        "version": MODEL_VERSION,
        "metrics": predictor.get_metrics(),
        "feature_importance": predictor.get_feature_importance(),
    }


@router.post("/retrain")
async def retrain():
    """Trigger a manual model retrain."""
    from src.db.session import async_session_factory
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    async with async_session_factory() as db:
        metrics = await trainer.train(db)
    return {"status": "complete", "metrics": metrics}
