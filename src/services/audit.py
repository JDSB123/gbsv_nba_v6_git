"""Data audit service — counts, breakdowns, recent activity.

Extracted from ``__main__._run_audit`` to keep the CLI thin and make
the logic independently testable and reusable (e.g. from API routes).
"""

from __future__ import annotations

from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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


async def run_audit(db: AsyncSession) -> None:
    """Execute the full data audit against *db* and print results."""

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
