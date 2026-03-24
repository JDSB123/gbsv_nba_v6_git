import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.__main__ import _setup_logging
from src.api.routes import health, model, odds, performance, predictions
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — use shared logging config (JSON in prod, plain in dev)
    _setup_logging()
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


# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.api_base_url] if settings.api_base_url else [],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ── Security headers ─────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── API key authentication ────────────────────────────────────────
@app.middleware("http")
async def api_key_auth(request: Request, call_next) -> Response:
    # Skip auth if no API key is configured or for health endpoints
    if not settings.api_key or request.url.path.startswith("/health") or request.url.path == "/docs" or request.url.path == "/openapi.json":
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != settings.api_key:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(odds.router)
app.include_router(model.router)
app.include_router(performance.router)
