from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func as sa_func
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_predictor
from src.db.models import Game, Injury, OddsSnapshot, Prediction, TeamSeasonStats
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

    # Injuries
    latest_injury = (
        await db.execute(select(sa_func.max(Injury.reported_at)))
    ).scalar_one_or_none()
    if latest_injury:
        age = (datetime.now(UTC) - latest_injury.replace(tzinfo=UTC)).total_seconds() / 60
        sources["injuries"] = {
            "latest": latest_injury.isoformat(),
            "age_minutes": round(age, 1),
            "status": "fresh" if age < 180 else "stale",
        }
    else:
        sources["injuries"] = {"status": "missing"}

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
        await db.execute(
            select(sa_func.count()).select_from(Game).where(Game.status == "NS")
        )
    ).scalar()
    ft_count = (
        await db.execute(
            select(sa_func.count()).select_from(Game).where(Game.status == "FT")
        )
    ).scalar()
    sources["games"] = {"upcoming": ns_count, "completed": ft_count}

    return sources
