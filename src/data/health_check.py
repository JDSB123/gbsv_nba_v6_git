"""Startup health check — validates every data source on worker boot.

Called once when the worker starts so silent failures are caught immediately
rather than hours later when a polling job finally runs.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)


async def check_odds_api() -> dict[str, Any]:
    """Ping The Odds API v4 /sports endpoint (free, 0 credits)."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.odds_api_base}/sports",
                params={"apiKey": settings.odds_api_key},
                timeout=15,
            )
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
            sports = resp.json()
            nba_found = any(s.get("key") == settings.odds_api_sport_key for s in sports)
            return {
                "source": "odds_api_v4",
                "status": "ok" if nba_found else "warning",
                "quota_remaining": int(remaining) if remaining else None,
                "quota_used": int(used) if used else None,
                "nba_active": nba_found,
                "regions": settings.odds_api_regions,
            }
    except Exception as exc:
        return {"source": "odds_api_v4", "status": "error", "detail": str(exc)}


async def check_basketball_api() -> dict[str, Any]:
    """Ping Basketball API v1 /status endpoint."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.basketball_api_base}/status",
                headers={"x-apisports-key": settings.basketball_api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            account = data.get("response", {}).get("account", {})
            requests_info = data.get("response", {}).get("requests", {})
            current = requests_info.get("current", 0)
            limit_day = requests_info.get("limit_day", 0)
            return {
                "source": "basketball_api_v1",
                "status": "ok",
                "plan": account.get("subscription", {}).get("plan", "unknown"),
                "requests_today": current,
                "daily_limit": limit_day,
                "pct_used": round(current / max(limit_day, 1) * 100, 1),
            }
    except Exception as exc:
        return {"source": "basketball_api_v1", "status": "error", "detail": str(exc)}


async def check_nba_api_v2() -> dict[str, Any]:
    """Ping NBA API v2 /status — always reports disabled when config says so."""
    settings = get_settings()
    if not settings.nba_api_v2_enabled:
        return {
            "source": "nba_api_v2",
            "status": "disabled",
            "detail": "NBA_API_V2_ENABLED=false — free tier exhausted, injuries/refs unavailable",
        }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.nba_api_base}/status",
                headers={"x-apisports-key": settings.basketball_api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            requests_info = data.get("response", {}).get("requests", {})
            current = requests_info.get("current", 0)
            limit_day = requests_info.get("limit_day", 0)
            exhausted = limit_day > 0 and current >= limit_day
            return {
                "source": "nba_api_v2",
                "status": "warning" if exhausted else "ok",
                "requests_today": current,
                "daily_limit": limit_day,
                "exhausted": exhausted,
            }
    except Exception as exc:
        return {"source": "nba_api_v2", "status": "error", "detail": str(exc)}


async def check_database() -> dict[str, Any]:
    """Verify database connectivity."""
    try:
        from sqlalchemy import text

        from src.db.session import async_session_factory

        async with async_session_factory() as db:
            await db.execute(text("SELECT 1"))
        return {"source": "database", "status": "ok"}
    except Exception as exc:
        return {"source": "database", "status": "error", "detail": str(exc)}


async def run_startup_health_check() -> list[dict[str, Any]]:
    """Run all health checks and log a startup status table.

    Returns the list of check results for use by API endpoints.
    """
    logger.info("=" * 60)
    logger.info("STARTUP HEALTH CHECK — validating all data sources")
    logger.info("=" * 60)

    results = []
    for check_fn in (check_database, check_odds_api, check_basketball_api, check_nba_api_v2):
        result = await check_fn()
        results.append(result)

    # Log status table
    for r in results:
        icon = {"ok": "✅", "warning": "⚠️", "error": "❌", "disabled": "🚫"}.get(
            r["status"], "?"
        )
        source = r["source"]
        detail_parts: list[str] = []
        if "quota_remaining" in r:
            detail_parts.append(f"quota={r['quota_remaining']:,}")
        if "regions" in r:
            detail_parts.append(f"regions={r['regions']}")
        if "plan" in r:
            detail_parts.append(f"plan={r['plan']}")
        if "pct_used" in r:
            detail_parts.append(f"used={r['pct_used']}%")
        if "exhausted" in r and r["exhausted"]:
            detail_parts.append("EXHAUSTED")
        if "detail" in r:
            detail_parts.append(r["detail"])
        detail = " | ".join(detail_parts) if detail_parts else ""
        logger.info("  %s  %-25s  %s", icon, source, detail)

    errors = [r for r in results if r["status"] == "error"]
    if errors:
        logger.error(
            "STARTUP CHECK: %d data source(s) FAILED — %s",
            len(errors),
            ", ".join(r["source"] for r in errors),
        )
    else:
        logger.info("STARTUP CHECK: All data sources healthy")

    logger.info("=" * 60)

    # Cache for API endpoint
    _last_check_results.clear()
    _last_check_results.extend(results)
    _last_check_time[0] = datetime.now(UTC).replace(tzinfo=None)

    return results


# Module-level cache so the /health/data-sources endpoint can return
# the latest check without re-running all probes.
_last_check_results: list[dict[str, Any]] = []
_last_check_time: list[Any] = [None]  # mutable container for timestamp


def get_last_check() -> tuple[list[dict[str, Any]], Any]:
    return _last_check_results, _last_check_time[0]
