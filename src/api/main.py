import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, model, odds, performance, predictions
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    logger.info("Starting NBA GBSV v6 — env=%s", settings.app_env)

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("API shutdown complete")


app = FastAPI(
    title="NBA GBSV v6 API",
    description="NBA prediction model for 6 markets (1H & FG spread/moneyline/total O/U)",
    version="6.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(odds.router)
app.include_router(model.router)
app.include_router(performance.router)
