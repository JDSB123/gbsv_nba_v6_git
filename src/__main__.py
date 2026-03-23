"""Single entry point for the NBA GBSV v6 system.

Usage:
    python -m src serve        # Start the FastAPI API server
    python -m src work         # Start the background scheduler (worker)
    python -m src train        # Run a one-shot model training
    python -m src predict      # Run predictions for all upcoming games (fresh odds)
    python -m src publish-teams # Generate predictions and publish to Teams
    python -m src backfill     # Backfill historical data from APIs
    python -m src migrate      # Run Alembic migrations (head)
"""

import argparse
import asyncio
import logging
import os

from src.config import get_settings
from src.data.seasons import current_nba_season


def _setup_logging(log_level: str | None = None) -> None:
    resolved_log_level = log_level
    if resolved_log_level is None:
        settings = get_settings()
        resolved_log_level = settings.log_level
    logging.basicConfig(
        level=getattr(logging, resolved_log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server via uvicorn."""
    _setup_logging()
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=(get_settings().app_env == "development"),
    )


def cmd_work(args: argparse.Namespace) -> None:
    """Start the background scheduler loop."""
    _setup_logging()
    logger = logging.getLogger("src.worker")
    logger.info("Starting worker (scheduler)...")

    async def _run() -> None:
        from src.data.scheduler import create_scheduler

        scheduler = create_scheduler()
        scheduler.start()
        logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            scheduler.shutdown(wait=False)
            logger.info("Worker shut down")

    asyncio.run(_run())


async def _run_train() -> None:
    from src.db.session import async_session_factory
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    async with async_session_factory() as db:
        metrics = await trainer.train(db)
    if metrics:
        print("Training complete. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Training skipped — not enough data.")


def cmd_train(args: argparse.Namespace) -> None:
    """One-shot model training."""
    _setup_logging()
    asyncio.run(_run_train())


async def _run_predict() -> None:
    from src.db.session import async_session_factory
    from src.models.predictor import Predictor

    predictor = Predictor()
    if not predictor.is_ready:
        print("Models not loaded. Run `python -m src train` first.")
        return
    async with async_session_factory() as db:
        predictions = await predictor.predict_upcoming(db)
    print(f"Generated {len(predictions)} predictions (fresh odds).")


def cmd_predict(args: argparse.Namespace) -> None:
    """Run predictions for all upcoming games using fresh odds."""
    _setup_logging()
    asyncio.run(_run_predict())


async def _run_publish_teams() -> None:
    from src.config import get_settings
    from src.data.scheduler import generate_predictions_and_publish

    settings = get_settings()
    has_graph = settings.teams_team_id and settings.teams_channel_id
    has_webhook = bool(settings.teams_webhook_url)
    if not has_graph and not has_webhook:
        print("Teams delivery not configured. Set TEAMS_WEBHOOK_URL or TEAMS_TEAM_ID + TEAMS_CHANNEL_ID.")
        return
    await generate_predictions_and_publish()
    print("Prediction publish job executed.")


def cmd_publish_teams(args: argparse.Namespace) -> None:
    """Generate predictions and publish formatted output to Teams."""
    _setup_logging()
    asyncio.run(_run_publish_teams())


async def _run_backfill(season: str | None, days_back: int) -> None:
    from src.data.backfill import run_backfill

    await run_backfill(season=season, days_back=days_back)


def cmd_backfill(args: argparse.Namespace) -> None:
    """Backfill historical data from APIs."""
    _setup_logging()
    asyncio.run(_run_backfill(season=args.season, days_back=args.days))


async def _run_sync() -> None:
    from src.data.scheduler import sync_events_to_games

    await sync_events_to_games()


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync Odds API events to games table (map odds_api_id)."""
    _setup_logging()
    asyncio.run(_run_sync())
    print("Sync complete.")


async def _run_odds() -> None:
    from src.data.scheduler import poll_fg_odds

    await poll_fg_odds()


def cmd_odds(args: argparse.Namespace) -> None:
    """Fetch and persist full-game odds."""
    _setup_logging()
    asyncio.run(_run_odds())
    print("Odds poll complete.")


def cmd_migrate(args: argparse.Namespace) -> None:
    """Run Alembic migrations to head."""
    _setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    print("Migrations applied.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="NBA GBSV v6 — single entry point",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    # work
    p_work = sub.add_parser("work", help="Start background scheduler")
    p_work.set_defaults(func=cmd_work)

    # train
    p_train = sub.add_parser("train", help="One-shot model training")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict upcoming games (fresh odds)")
    p_pred.set_defaults(func=cmd_predict)

    # publish-teams
    p_pub = sub.add_parser(
        "publish-teams", help="Generate predictions and publish to Teams"
    )
    p_pub.set_defaults(func=cmd_publish_teams)

    # backfill
    p_back = sub.add_parser("backfill", help="Backfill historical data")
    p_back.add_argument(
        "--season",
        default=None,
        help=f"Season to backfill (default: {current_nba_season()})",
    )
    p_back.add_argument("--days", type=int, default=90, help="Days of game history")
    p_back.set_defaults(func=cmd_backfill)

    # migrate
    p_mig = sub.add_parser("migrate", help="Run Alembic migrations")
    p_mig.set_defaults(func=cmd_migrate)

    # sync
    p_sync = sub.add_parser("sync", help="Sync Odds API events to games")
    p_sync.set_defaults(func=cmd_sync)

    # odds
    p_odds = sub.add_parser("odds", help="Fetch and persist full-game odds")
    p_odds.set_defaults(func=cmd_odds)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
