"""Dead-letter queue retry job — retries failed ingestion records with exponential backoff."""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from src.db.models import IngestionFailure
from src.db.session import async_session_factory

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF_MINUTES = 10  # quadratic: 10, 40, 90, 160 min from failed_at


async def process_dead_letter_queue() -> int:
    """Retry failed ingestion records.  Returns count of records retried."""
    retried = 0
    try:
        async with async_session_factory() as db:
            now = datetime.now(UTC).replace(tzinfo=None)
            result = await db.execute(
                select(IngestionFailure)
                .where(
                    IngestionFailure.retry_count < _MAX_RETRIES,
                    IngestionFailure.permanently_failed.is_(False),
                    IngestionFailure.resolved_at.is_(None),
                )
                .order_by(IngestionFailure.failed_at)
                .limit(20)
            )
            failures = result.scalars().all()

            for failure in failures:
                # Exponential backoff: skip if not enough time has passed
                backoff_minutes = _BASE_BACKOFF_MINUTES * ((failure.retry_count + 1) ** 2)
                next_retry_at = failure.failed_at + timedelta(minutes=backoff_minutes)
                if now < next_retry_at:
                    continue

                logger.info(
                    "Retrying DLQ record id=%s job=%s attempt=%d",
                    failure.id,
                    failure.job_name,
                    failure.retry_count + 1,
                )

                try:
                    job_fn = _resolve_job(failure.job_name)
                    if job_fn is None:
                        logger.warning("Unknown job name in DLQ: %s", failure.job_name)
                        failure.permanently_failed = True
                        continue

                    await job_fn()
                    # Success — mark resolved (terminal state, won't be selected again)
                    failure.resolved_at = now
                    retried += 1
                    logger.info("DLQ retry succeeded for id=%s job=%s", failure.id, failure.job_name)

                except Exception as exc:
                    failure.retry_count = failure.retry_count + 1
                    failure.error_message = str(exc)[:2000]
                    if failure.retry_count >= _MAX_RETRIES:
                        failure.permanently_failed = True
                        logger.warning(
                            "DLQ record id=%s permanently failed after %d retries",
                            failure.id,
                            failure.retry_count,
                        )

            await db.commit()

        if retried:
            logger.info("DLQ processing complete: %d records retried successfully", retried)
    except Exception:
        logger.exception("Error processing dead-letter queue")

    return retried


def _resolve_job(job_name: str):
    """Map a job name string to its async callable."""
    from src.data.jobs.polling import (
        poll_1h_odds,
        poll_fg_odds,
        poll_injuries,
        poll_player_props,
        poll_scores_and_box,
        poll_stats,
    )

    registry = {
        "poll_fg_odds": poll_fg_odds,
        "poll_1h_odds": poll_1h_odds,
        "poll_player_props": poll_player_props,
        "poll_stats": poll_stats,
        "poll_scores_and_box": poll_scores_and_box,
        "poll_injuries": poll_injuries,
    }
    return registry.get(job_name)
