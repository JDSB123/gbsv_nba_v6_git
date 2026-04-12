from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func as sa_func
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_predictor
from src.config import get_settings
from src.db.models import Game, OddsSnapshot, Prediction, TeamSeasonStats
from src.db.session import get_db
from src.models.predictor import Predictor

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/deep")
async def health_deep(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Deep health check: verifies DB, models, and data freshness."""
    checks: dict = {}
    overall = "ok"

    # 1. Database connectivity
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)}
        overall = "degraded"

    # 2. Model readiness
    if predictor.is_ready:
        status_info = predictor.get_runtime_status()
        checks["models"] = {
            "status": "ok",
            "version": status_info.get("model_version"),
            "compatibility_mode": status_info.get("compatibility_mode", False),
        }
    else:
        checks["models"] = {"status": "error", "detail": "Models not loaded"}
        overall = "degraded"

    # 3. Odds data freshness
    try:
        latest_odds = (
            await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
        ).scalar_one_or_none()
        if latest_odds is not None:
            age_minutes = (datetime.now(UTC) - latest_odds.replace(tzinfo=UTC)).total_seconds() / 60
            checks["odds_freshness"] = {
                "status": "ok" if age_minutes < 60 else "warning",
                "latest_captured_at": latest_odds.isoformat(),
                "age_minutes": round(age_minutes, 1),
            }
        else:
            checks["odds_freshness"] = {"status": "warning", "detail": "No odds data"}
            overall = "degraded"
    except Exception as exc:
        checks["odds_freshness"] = {"status": "error", "detail": str(exc)}

    # 4. Team stats coverage
    try:
        team_count = (
            await db.execute(select(sa_func.count()).select_from(TeamSeasonStats))
        ).scalar()
        checks["team_stats"] = {
            "status": "ok" if team_count and team_count >= 30 else "warning",
            "teams_with_stats": team_count,
        }
    except Exception as exc:
        checks["team_stats"] = {"status": "error", "detail": str(exc)}

    return {"status": overall, "checks": checks}


@router.get("/health/freshness")
async def health_freshness(db: AsyncSession = Depends(get_db)):
    """Data freshness dashboard — latest timestamps for each data source."""
    sources: dict = {}

    # Odds snapshots
    latest_odds = (
        await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
    ).scalar_one_or_none()
    if latest_odds:
        age = (datetime.now(UTC) - latest_odds.replace(tzinfo=UTC)).total_seconds() / 60
        sources["odds"] = {
            "latest": latest_odds.isoformat(),
            "age_minutes": round(age, 1),
            "status": "fresh" if age < 30 else "stale" if age < 120 else "very_stale",
        }
    else:
        sources["odds"] = {"status": "missing"}

    # Latest prediction
    latest_pred = (
        await db.execute(select(sa_func.max(Prediction.predicted_at)))
    ).scalar_one_or_none()
    if latest_pred:
        age = (datetime.now(UTC) - latest_pred.replace(tzinfo=UTC)).total_seconds() / 60
        sources["predictions"] = {
            "latest": latest_pred.isoformat(),
            "age_minutes": round(age, 1),
        }
    else:
        sources["predictions"] = {"status": "missing"}

    # Games coverage
    ns_count = (
        await db.execute(select(sa_func.count()).select_from(Game).where(Game.status == "NS"))
    ).scalar()
    ft_count = (
        await db.execute(select(sa_func.count()).select_from(Game).where(Game.status == "FT"))
    ).scalar()
    sources["games"] = {"upcoming": ns_count, "completed": ft_count}

    return sources


@router.get("/health/data-sources")
async def health_data_sources(db: AsyncSession = Depends(get_db)):
    """Live data-source status — single source of truth for pipeline health.

    Returns cached startup-check results plus real-time DB freshness
    and the active configuration (regions, markets, intervals).
    """
    from src.data.health_check import get_last_check, run_startup_health_check

    settings = get_settings()

    # Re-run checks if never run (API-only mode, no worker) or stale (>5 min)
    cached, checked_at = get_last_check()
    if not cached or (
        checked_at and (datetime.now(UTC).replace(tzinfo=None) - checked_at).total_seconds() > 300
    ):
        cached = await run_startup_health_check()
        checked_at = datetime.now(UTC).replace(tzinfo=None)

    # Enrich with DB freshness
    latest_odds = (
        await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
    ).scalar_one_or_none()
    odds_age = None
    if latest_odds:
        odds_age = round(
            (datetime.now(UTC) - latest_odds.replace(tzinfo=UTC)).total_seconds() / 60, 1
        )

    # Distinct bookmakers in last 24h
    from sqlalchemy import distinct

    bk_result = await db.execute(
        select(distinct(OddsSnapshot.bookmaker)).where(
            OddsSnapshot.captured_at >= datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24)
        )
    )
    active_bookmakers = sorted([row[0] for row in bk_result.fetchall()])

    return {
        "checked_at": checked_at.isoformat() if checked_at else None,
        "api_checks": cached,
        "config": {
            "regions": settings.odds_api_regions,
            "markets_fg": settings.odds_api_markets_fg,
            "markets_1h": settings.odds_api_markets_1h,
            "odds_fg_interval_min": settings.odds_fg_interval,
            "odds_1h_interval_min": settings.odds_1h_interval,
        },
        "freshness": {
            "odds_latest": latest_odds.isoformat() if latest_odds else None,
            "odds_age_minutes": odds_age,
            "odds_status": (
                "fresh"
                if odds_age and odds_age < 30
                else "stale"
                if odds_age and odds_age < 120
                else "very_stale"
                if odds_age
                else "missing"
            ),
        },
        "active_bookmakers_24h": active_bookmakers,
        "active_bookmaker_count": len(active_bookmakers),
    }
