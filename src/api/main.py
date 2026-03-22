import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, predictions
from src.config import get_settings
from src.data.scheduler import create_scheduler

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info("Starting NBA GBSV v6 — env=%s", settings.app_env)

    scheduler = create_scheduler()
    scheduler.start()
    logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


app = FastAPI(
    title="NBA GBSV v6 API",
    description="NBA prediction model for 6 markets (1H & FG spread/moneyline/total O/U)",
    version="6.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predictions.router)


@app.get("/odds/latest")
async def latest_odds():
    """Return latest cached odds snapshots."""
    from sqlalchemy import select
    from src.db.session import async_session_factory
    from src.db.models import Game, OddsSnapshot

    async with async_session_factory() as db:
        result = await db.execute(
            select(OddsSnapshot)
            .join(Game)
            .where(Game.status == "NS")
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(500)
        )
        snapshots = result.scalars().all()
        return {
            "odds": [
                {
                    "game_id": s.game_id,
                    "bookmaker": s.bookmaker,
                    "market": s.market,
                    "outcome": s.outcome_name,
                    "price": s.price,
                    "point": s.point,
                    "captured_at": s.captured_at.isoformat(),
                }
                for s in snapshots
            ],
            "count": len(snapshots),
        }


@app.get("/model/status")
async def model_status():
    """Return model version, metrics, and feature importance."""
    from src.api.dependencies import predictor

    return {
        "ready": predictor.is_ready,
        "version": "v6.0.0",
        "metrics": predictor.get_metrics(),
        "feature_importance": predictor.get_feature_importance(),
    }


@app.post("/model/retrain")
async def retrain():
    """Trigger a manual model retrain."""
    from src.db.session import async_session_factory
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    async with async_session_factory() as db:
        metrics = await trainer.train(db)
    return {"status": "complete", "metrics": metrics}
