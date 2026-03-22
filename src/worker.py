"""Standalone worker process — runs the scheduler in a loop."""

import asyncio
import logging

from src.config import get_settings
from src.data.scheduler import create_scheduler

settings = get_settings()


def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info("Starting worker process (scheduler only)")

    scheduler = create_scheduler()
    scheduler.start()

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Worker shut down")


if __name__ == "__main__":
    main()
