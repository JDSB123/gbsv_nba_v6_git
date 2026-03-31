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
from typing import Any

from src.config import get_settings
from src.data.seasons import current_nba_season


def _setup_logging(log_level: str | None = None) -> None:
    resolved_log_level = log_level
    if resolved_log_level is None:
        settings = get_settings()
        resolved_log_level = settings.log_level

    level = getattr(logging, resolved_log_level.upper(), logging.INFO)

    app_env = os.getenv("APP_ENV", "development")
    if app_env not in ("development", "test"):
        # Structured JSON logging for production
        from pythonjsonlogger.json import JsonFormatter

        handler = logging.StreamHandler()
        handler.setFormatter(
            JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
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
        import signal

        from src.data.scheduler import create_scheduler

        scheduler = create_scheduler()
        scheduler.start()
        logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))

        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Received shutdown signal, waiting for in-flight jobs...")
            stop_event.set()

        loop = asyncio.get_running_loop()
        import contextlib

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _signal_handler)

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            scheduler.shutdown(wait=True)
            logger.info("Worker shut down gracefully")

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


async def _summarize_upcoming_coverage(db: Any) -> dict[str, Any]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from src.db.models import Game

    result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.status == "NS")
        .order_by(Game.commence_time)
    )
    games = result.scalars().all()

    linked_games = [game for game in games if getattr(game, "odds_api_id", None)]
    awaiting_odds_games = []
    for game in games:
        if getattr(game, "odds_api_id", None):
            continue
        away_name = getattr(getattr(game, "away_team", None), "name", "?")
        home_name = getattr(getattr(game, "home_team", None), "name", "?")
        commence_time = getattr(game, "commence_time", None)
        when = commence_time.isoformat() if commence_time is not None else "unknown time"
        awaiting_odds_games.append(f"{away_name} @ {home_name} ({when})")

    return {
        "ns_game_count": len(games),
        "linked_ns_game_count": len(linked_games),
        "awaiting_odds_games": awaiting_odds_games,
    }


async def _run_predict() -> None:
    from src.data.scheduler import (
        purge_invalid_upcoming_predictions,
        poll_1h_odds,
        poll_fg_odds,
        poll_injuries,
        poll_player_props,
        poll_scores_and_box,
        poll_stats,
        sync_events_to_games,
    )
    from src.db.session import async_session_factory
    from src.models.predictor import Predictor

    await poll_stats()
    await poll_scores_and_box()
    await poll_injuries()
    await sync_events_to_games()
    await poll_fg_odds()
    await poll_1h_odds()
    await poll_player_props()

    async with async_session_factory() as db:
        purged_count = await purge_invalid_upcoming_predictions(db)
        if purged_count:
            print(f"Purged {purged_count} malformed upcoming predictions before refresh.")
        predictor = Predictor()
        if not predictor.is_ready:
            print("Models not loaded. Run `python -m src train` first.")
            return
        coverage = await _summarize_upcoming_coverage(db)
        predictions = await predictor.predict_upcoming(db)
    linked_ns_game_count = int(coverage["linked_ns_game_count"])
    awaiting_odds_games = list(coverage["awaiting_odds_games"])
    ns_game_count = int(coverage["ns_game_count"])

    print(
        f"Generated {len(predictions)} predictions for "
        f"{linked_ns_game_count} odds-linked upcoming games."
    )
    if len(predictions) < linked_ns_game_count:
        print(
            f"Eligible coverage incomplete: {len(predictions)} / "
            f"{linked_ns_game_count} odds-linked games predicted."
        )
    if awaiting_odds_games:
        print(
            f"Waiting on odds coverage for {len(awaiting_odds_games)} / "
            f"{ns_game_count} upcoming games:"
        )
        for summary in awaiting_odds_games[:10]:
            print(f"  {summary}")


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
    published_count = await generate_predictions_and_publish()
    print(f"Prediction publish job executed. Published {published_count} predictions.")


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


async def _run_perf() -> None:
    import json

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from src.api.routes.performance import _build_stats, _clv_summary, _grade_game, _score_accuracy
    from src.db.models import Game, Prediction
    from src.db.session import async_session_factory

    async with async_session_factory() as db:
        result = await db.execute(
            select(Prediction, Game)
            .join(Game, Prediction.game_id == Game.id)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.status == "FT")
            .where(Game.home_score_fg.isnot(None))
            .where(Game.away_score_fg.isnot(None))
            .order_by(Game.commence_time)
        )
        rows = result.all()

    if not rows:
        print("No completed games with predictions yet.")
        return

    from typing import Any, cast

    seen: dict[int, tuple[Any, Any]] = {}
    for pred, game in rows:
        gid = int(cast(Any, game.id))
        existing = seen.get(gid)
        if existing is None or pred.predicted_at > existing[0].predicted_at:
            seen[gid] = (pred, game)
    unique = list(seen.values())

    graded = []
    for pred, game in unique:
        graded.extend(_grade_game(pred, game))

    data = {
        "games_graded": len(unique),
        "picks_graded": len(graded),
        "accuracy": _score_accuracy(unique),
        "pick_performance": _build_stats(graded),
        "clv": _clv_summary(unique),
    }
    print(json.dumps(data, indent=2))


def cmd_perf(args: argparse.Namespace) -> None:
    """Show performance metrics for completed game predictions."""
    _setup_logging()
    asyncio.run(_run_perf())


def cmd_migrate(args: argparse.Namespace) -> None:
    """Run Alembic migrations to head."""
    _setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    print("Migrations applied.")


async def _run_audit() -> None:
    from sqlalchemy import func as sa_func
    from sqlalchemy import select

    from src.db.models import (
        Game,
        Injury,
        ModelRegistry,
        OddsSnapshot,
        Player,
        PlayerGameStats,
        Prediction,
        Team,
        TeamSeasonStats,
    )
    from src.db.session import async_session_factory

    async with async_session_factory() as db:
        # ── Table counts ──────────────────────────────────
        tables = [
            ("teams", Team),
            ("players", Player),
            ("games", Game),
            ("team_season_stats", TeamSeasonStats),
            ("player_game_stats", PlayerGameStats),
            ("odds_snapshots", OddsSnapshot),
            ("predictions", Prediction),
            ("model_registry", ModelRegistry),
            ("injuries", Injury),
        ]
        print("=" * 60)
        print("  DATA AUDIT — NBA GBSV v6")
        print("=" * 60)
        print()
        print("TABLE COUNTS:")
        for label, model in tables:
            cnt = (await db.execute(select(sa_func.count()).select_from(model))).scalar()
            print(f"  {label:<25} {cnt:>8,}")

        # ── Games breakdown by status ─────────────────────
        print()
        print("GAMES BY STATUS:")
        rows = (await db.execute(
            select(Game.status, sa_func.count())
            .group_by(Game.status)
            .order_by(sa_func.count().desc())
        )).all()
        for status, cnt in rows:
            print(f"  {status or 'NULL':<10} {cnt:>6,}")

        # ── Games by season ──────────────────────────────
        print()
        print("GAMES BY SEASON:")
        rows = (await db.execute(
            select(Game.season, sa_func.count())
            .group_by(Game.season)
            .order_by(Game.season)
        )).all()
        for season, cnt in rows:
            print(f"  {season or 'NULL':<15} {cnt:>6,}")

        # ── Games with odds_api_id ────────────────────────
        with_odds_id = (await db.execute(
            select(sa_func.count()).select_from(Game)
            .where(Game.odds_api_id.isnot(None))
        )).scalar()
        total_games = (await db.execute(
            select(sa_func.count()).select_from(Game)
        )).scalar()
        print()
        print(f"GAMES WITH odds_api_id:  {with_odds_id:,} / {total_games:,}")

        # ── Games with scores ─────────────────────────────
        with_scores = (await db.execute(
            select(sa_func.count()).select_from(Game)
            .where(Game.home_score_fg.isnot(None))
        )).scalar()
        with_1h = (await db.execute(
            select(sa_func.count()).select_from(Game)
            .where(Game.home_score_1h.isnot(None))
        )).scalar()
        print(f"GAMES WITH full scores:  {with_scores:,} / {total_games:,}")
        print(f"GAMES WITH 1H scores:    {with_1h:,} / {total_games:,}")

        # ── Predictions breakdown ─────────────────────────
        print()
        print("PREDICTIONS:")
        pred_count = (await db.execute(
            select(sa_func.count()).select_from(Prediction)
        )).scalar()
        pred_with_opening = (await db.execute(
            select(sa_func.count()).select_from(Prediction)
            .where(Prediction.opening_spread.isnot(None))
        )).scalar()
        pred_with_clv = (await db.execute(
            select(sa_func.count()).select_from(Prediction)
            .where(Prediction.clv_spread.isnot(None))
        )).scalar()
        pred_with_odds = (await db.execute(
            select(sa_func.count()).select_from(Prediction)
            .where(Prediction.odds_sourced.isnot(None))
        )).scalar()
        print(f"  Total predictions:       {pred_count:>6,}")
        print(f"  With opening lines:      {pred_with_opening:>6,}")
        print(f"  With CLV filled:         {pred_with_clv:>6,}")
        print(f"  With odds_sourced JSON:  {pred_with_odds:>6,}")

        # ── Predictions by model version ──────────────────
        rows = (await db.execute(
            select(Prediction.model_version, sa_func.count())
            .group_by(Prediction.model_version)
            .order_by(sa_func.count().desc())
        )).all()
        if rows:
            print()
            print("  BY MODEL VERSION:")
            for ver, cnt in rows:
                print(f"    {ver:<20} {cnt:>6,}")

        # ── Odds snapshots breakdown ──────────────────────
        print()
        print("ODDS SNAPSHOTS:")
        odds_count = (await db.execute(
            select(sa_func.count()).select_from(OddsSnapshot)
        )).scalar()
        unique_games_with_odds = (await db.execute(
            select(sa_func.count(OddsSnapshot.game_id.distinct()))
        )).scalar()
        print(f"  Total snapshots:         {odds_count:>6,}")
        print(f"  Unique games w/ odds:    {unique_games_with_odds:>6,}")

        # By market type
        rows = (await db.execute(
            select(OddsSnapshot.market, sa_func.count())
            .group_by(OddsSnapshot.market)
            .order_by(sa_func.count().desc())
        )).all()
        if rows:
            print("  BY MARKET:")
            for mkt, cnt in rows:
                print(f"    {mkt:<20} {cnt:>6,}")

        # Unique bookmakers
        rows = (await db.execute(
            select(OddsSnapshot.bookmaker, sa_func.count())
            .group_by(OddsSnapshot.bookmaker)
            .order_by(sa_func.count().desc())
        )).all()
        if rows:
            print(f"  UNIQUE BOOKMAKERS: {len(rows)}")
            for bk, cnt in rows:
                print(f"    {bk:<25} {cnt:>6,}")

        # Date range of odds
        oldest = (await db.execute(select(sa_func.min(OddsSnapshot.captured_at)))).scalar()
        newest = (await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))).scalar()
        if oldest and newest:
            print(f"  DATE RANGE: {oldest} → {newest}")

        # ── Model registry ────────────────────────────────
        print()
        print("MODEL REGISTRY:")
        mr_rows = (await db.execute(
            select(ModelRegistry)
            .order_by(ModelRegistry.created_at.desc())
            .limit(5)
        )).scalars().all()
        if mr_rows:
            for m in mr_rows:
                active = "ACTIVE" if m.is_active else "retired"
                print(f"  {m.model_version:<20} {active:<10} created={m.created_at}")
        else:
            print("  (empty)")

        # ── Recent predictions detail ─────────────────────
        print()
        print("RECENT PREDICTIONS (last 10):")
        from sqlalchemy.orm import selectinload
        p_rows = (await db.execute(
            select(Prediction)
            .options(selectinload(Prediction.game).selectinload(Game.home_team),
                     selectinload(Prediction.game).selectinload(Game.away_team))
            .order_by(Prediction.predicted_at.desc())
            .limit(10)
        )).scalars().all()
        for p in p_rows:
            g = p.game
            home = g.home_team.name if g and g.home_team else "?"
            away = g.away_team.name if g and g.away_team else "?"
            status = g.status if g else "?"
            o_spr = f"{p.opening_spread:+.1f}" if p.opening_spread is not None else "none"
            o_tot = f"{p.opening_total:.1f}" if p.opening_total is not None else "none"
            clv = f"CLV={p.clv_spread:+.1f}" if p.clv_spread is not None else "no CLV"
            actual = ""
            if g and g.home_score_fg is not None:
                actual = f" actual={g.home_score_fg}-{g.away_score_fg}"
            print(f"  {away} @ {home} [{status}] pred={p.predicted_home_fg:.0f}-{p.predicted_away_fg:.0f}"
                  f"  spread={o_spr} total={o_tot} {clv}{actual}")

        # ── Upcoming games (NS) ──────────────────────────
        print()
        print("UPCOMING GAMES (NS):")
        g_rows = (await db.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.status == "NS")
            .order_by(Game.commence_time)
            .limit(15)
        )).scalars().all()
        for g in g_rows:
            home = g.home_team.name if g.home_team else "?"
            away = g.away_team.name if g.away_team else "?"
            oid = g.odds_api_id or "NO_OID"
            print(f"  {g.commence_time} | {away} @ {home} | {oid}")

        print()
        print("=" * 60)
        print("  AUDIT COMPLETE")
        print("=" * 60)


def cmd_audit(args: argparse.Namespace) -> None:
    """Full data audit — counts, breakdowns, recent activity."""
    _setup_logging()
    asyncio.run(_run_audit())


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

    # perf
    p_perf = sub.add_parser("perf", help="Show performance metrics")
    p_perf.set_defaults(func=cmd_perf)

    # audit
    p_audit = sub.add_parser("audit", help="Full data audit and counts")
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
