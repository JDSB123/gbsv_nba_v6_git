"""Admin routes — scheduler status, job triggers, model promotion."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.rate_limit import limiter
from src.config import get_settings
from src.db.models import ModelAuditLog, ModelRegistry
from src.db.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.api_key:
        raise HTTPException(status_code=403, detail="API key not configured on server")
    provided = request.headers.get("X-API-Key", "")
    if provided != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Scheduler status ─────────────────────────────────────────────


@router.get("/scheduler/status")
async def scheduler_status(
    _auth: None = Depends(_require_api_key),
) -> dict[str, Any]:
    """Return APScheduler job states: next run, last error, etc."""
    try:
        from src.data.scheduler import _scheduler_instance

        scheduler = _scheduler_instance
        if scheduler is None:
            return {"status": "not_running", "jobs": []}

        jobs = []
        for job in scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run.isoformat() if next_run else None,
                    "trigger": str(job.trigger),
                }
            )
        return {"status": "running", "jobs": jobs}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ── Job trigger ──────────────────────────────────────────────────


_JOB_REGISTRY: dict[str, Any] = {}


def _get_job_registry() -> dict[str, Any]:
    if _JOB_REGISTRY:
        return _JOB_REGISTRY

    from src.data.jobs.dead_letter import process_dead_letter_queue
    from src.data.jobs.maintenance import db_maintenance, fill_clv, prune_old_odds
    from src.data.jobs.polling import (
        daily_retrain,
        poll_1h_odds,
        poll_fg_odds,
        poll_player_props,
        poll_scores_and_box,
        poll_stats,
        sync_events_to_games,
    )
    from src.data.jobs.predictions import (
        generate_predictions_and_publish,
        pregame_check,
        refresh_prediction_cache,
    )

    _JOB_REGISTRY.update(
        {
            "poll_fg_odds": poll_fg_odds,
            "poll_1h_odds": poll_1h_odds,
            "poll_player_props": poll_player_props,
            "poll_stats": poll_stats,
            "poll_scores_and_box": poll_scores_and_box,
            "sync_events_to_games": sync_events_to_games,
            "daily_retrain": daily_retrain,
            "fill_clv": fill_clv,
            "prune_old_odds": prune_old_odds,
            "db_maintenance": db_maintenance,
            "generate_predictions_and_publish": generate_predictions_and_publish,
            "pregame_check": pregame_check,
            "refresh_prediction_cache": refresh_prediction_cache,
            "process_dead_letter_queue": process_dead_letter_queue,
        }
    )
    return _JOB_REGISTRY


@router.post("/jobs/trigger")
@limiter.limit("10/minute")
async def trigger_job(
    request: Request,
    job_name: str,
    _auth: None = Depends(_require_api_key),
) -> dict[str, Any]:
    """Manually trigger a scheduler job by name."""
    registry = _get_job_registry()
    if job_name not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job: {job_name}. Available: {sorted(registry.keys())}",
        )
    fn = registry[job_name]
    result = await fn()
    return {"job": job_name, "status": "triggered", "result": str(result)}


# ── Model promotion ─────────────────────────────────────────────


@router.post("/model/promote")
@limiter.limit("5/minute")
async def promote_model(
    request: Request,
    version: str,
    reason: str = "manual promotion",
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> dict[str, Any]:
    """Promote a specific model version to active, with audit trail."""
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.model_version == version)
    )
    target = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail=f"Model version {version!r} not found")

    now = datetime.now(UTC).replace(tzinfo=None)

    # Retire currently active models
    active_result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.is_active.is_(True),
            ModelRegistry.model_version != version,
        )
    )
    previous_version = None
    for row in active_result.scalars().all():
        previous_version = row.model_version
        row.is_active = False
        row.retired_at = now

    # Promote target
    target.is_active = True
    target.promoted_at = now
    target.retired_at = None
    target.promotion_reason = reason

    # Audit log
    db.add(
        ModelAuditLog(
            model_version=version,
            action="promote",
            previous_version=previous_version,
            reason=reason,
        )
    )

    await db.commit()

    return {
        "status": "promoted",
        "version": version,
        "previous_version": previous_version,
        "reason": reason,
    }


@router.post("/model/rollback")
@limiter.limit("5/minute")
async def rollback_model(
    request: Request,
    version: str,
    reason: str = "manual rollback",
    db: AsyncSession = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> dict[str, Any]:
    """Roll back to a previous model version."""
    # Delegate to promote with rollback action logged
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.model_version == version)
    )
    target = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail=f"Model version {version!r} not found")

    now = datetime.now(UTC).replace(tzinfo=None)

    active_result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.is_active.is_(True),
            ModelRegistry.model_version != version,
        )
    )
    previous_version = None
    for row in active_result.scalars().all():
        previous_version = row.model_version
        row.is_active = False
        row.retired_at = now

    target.is_active = True
    target.promoted_at = now
    target.retired_at = None
    target.promotion_reason = f"rollback: {reason}"

    db.add(
        ModelAuditLog(
            model_version=version,
            action="rollback",
            previous_version=previous_version,
            reason=reason,
        )
    )

    await db.commit()

    return {
        "status": "rolled_back",
        "version": version,
        "previous_version": previous_version,
        "reason": reason,
    }


@router.get("/model/audit-log")
async def model_audit_log(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    _auth: None = Depends(_require_api_key),
) -> dict[str, Any]:
    """Return model promotion/rollback audit trail."""
    result = await db.execute(
        select(ModelAuditLog)
        .order_by(ModelAuditLog.performed_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return {
        "entries": [
            {
                "id": row.id,
                "model_version": row.model_version,
                "action": row.action,
                "previous_version": row.previous_version,
                "reason": row.reason,
                "performed_at": row.performed_at.isoformat() if row.performed_at else None,
            }
            for row in rows
        ]
    }
