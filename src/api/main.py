import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.__main__ import _setup_logging
from src.api.rate_limit import limiter
from src.api.routes import admin, health, model, odds, performance, predictions
from src.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — use shared logging config (JSON in prod, plain in dev)
    _setup_logging()
    settings = get_settings()

    # ── Application Insights (OpenTelemetry) ──────────────────────
    if settings.applicationinsights_connection_string:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(
                logger_name="src",
                instrumentation_options={"fastapi": {"enabled": True}},
            )
            logger.info("Application Insights telemetry enabled")
        except Exception:
            logger.warning("Failed to configure Application Insights", exc_info=True)

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

# ── Rate limiting ────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── CORS ──────────────────────────────────────────────────────────
_cors_settings = get_settings()
_cors_origins: list[str] = list(_cors_settings.cors_origins)
if _cors_settings.api_base_url and _cors_settings.api_base_url not in _cors_origins:
    _cors_origins.append(_cors_settings.api_base_url)
if _cors_settings.app_env == "development" and not _cors_origins:
    _cors_origins += [
        f"http://localhost:{_cors_settings.server_port}",
        f"http://127.0.0.1:{_cors_settings.server_port}",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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
_AUTH_EXEMPT = {
    "/",
    "/health",
    "/health/deep",
    "/health/freshness",
    "/model/status",
    "/docs",
    "/openapi.json",
}


@app.middleware("http")
async def api_key_auth(request: Request, call_next) -> Response:
    settings = get_settings()
    if not settings.api_key or request.url.path in _AUTH_EXEMPT:
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != settings.api_key:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


@app.get("/", include_in_schema=False)
async def root(request: Request) -> HTMLResponse:
    base = str(request.base_url).rstrip("/")
    settings = get_settings()
    auth_note = (
        "Prediction endpoints may require an X-API-Key on this deployment."
        if settings.api_key
        else "Prediction endpoints are currently reachable without an API key."
    )
    links = [
        ("API Docs", f"{base}/docs"),
        ("Health", f"{base}/health"),
        ("Deep Health", f"{base}/health/deep"),
        ("Freshness", f"{base}/health/freshness"),
        ("Predictions JSON", f"{base}/predictions"),
        ("Slate HTML", f"{base}/predictions/slate.html"),
        ("Slate CSV", f"{base}/predictions/slate.csv"),
    ]
    items = "".join(
        f'<li style="margin:8px 0"><a href="{href}" style="color:#0f62fe;text-decoration:none">{label}</a></li>'
        for label, href in links
    )
    html = (
        "<!DOCTYPE html><html><head><title>NBA GBSV v6 API</title></head>"
        '<body style="margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f7fb;color:#14213d">'
        '<main style="max-width:760px;margin:48px auto;padding:32px;background:#ffffff;'
        'border:1px solid #dbe3f0;border-radius:18px;box-shadow:0 18px 40px rgba(20,33,61,.08)">'
        '<div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#4f6d8a;'
        'font-weight:700">NBA GBSV v6</div>'
        '<h1 style="margin:10px 0 12px;font-size:32px;line-height:1.1">Live API is running</h1>'
        '<p style="margin:0 0 18px;font-size:15px;line-height:1.6;color:#40566f">'
        "This service exposes health checks, prediction endpoints, and a browser-friendly slate view."
        "</p>"
        f'<p style="margin:0 0 20px;font-size:14px;line-height:1.6;color:#5b728a">{auth_note}</p>'
        '<ul style="margin:0;padding-left:20px;font-size:15px;line-height:1.6">'
        f"{items}</ul>"
        "</main></body></html>"
    )
    return HTMLResponse(content=html)


app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(odds.router)
app.include_router(model.router)
app.include_router(performance.router)
app.include_router(admin.router)
