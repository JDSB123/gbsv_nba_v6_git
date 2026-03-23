from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import get_predictor
from src.config import get_settings
from src.db.models import Game, OddsSnapshot, Prediction
from src.db.session import get_db
from src.models.predictor import Predictor
from src.notifications.teams import (
    build_html_slate,
    build_slate_csv,
    build_teams_card,
    send_card_to_teams,
    send_card_via_graph,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])
settings = get_settings()


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _format_prediction(pred: Prediction, game: Game) -> dict:
    home_name = (
        game.home_team.name
        if game.home_team is not None
        else f"Team {game.home_team_id}"
    )
    away_name = (
        game.away_team.name
        if game.away_team is not None
        else f"Team {game.away_team_id}"
    )
    fg_home_ml_prob = _as_float(pred.fg_home_ml_prob, 0.5)
    h1_home_ml_prob = _as_float(pred.h1_home_ml_prob, 0.5)
    return {
        "game_id": game.id,
        "odds_api_id": game.odds_api_id,
        "commence_time": (
            game.commence_time.isoformat() if game.commence_time is not None else None
        ),
        "away_team": away_name,
        "home_team": home_name,
        "predicted_scores": {
            "full_game": {
                "away": pred.predicted_away_fg,
                "home": pred.predicted_home_fg,
            },
            "first_half": {
                "away": pred.predicted_away_1h,
                "home": pred.predicted_home_1h,
            },
        },
        "markets": {
            "fg_spread": {"prediction": pred.fg_spread},
            "fg_total": {"prediction": pred.fg_total},
            "fg_moneyline": {
                "home_prob": fg_home_ml_prob,
                "away_prob": round(1 - fg_home_ml_prob, 3),
            },
            "h1_spread": {"prediction": pred.h1_spread},
            "h1_total": {"prediction": pred.h1_total},
            "h1_moneyline": {
                "home_prob": h1_home_ml_prob,
                "away_prob": round(1 - h1_home_ml_prob, 3),
            },
        },
        "model_version": pred.model_version,
        "predicted_at": (
            pred.predicted_at.isoformat() if pred.predicted_at is not None else None
        ),
        "clv": {
            "opening_spread": pred.opening_spread,
            "opening_total": pred.opening_total,
            "closing_spread": pred.closing_spread,
            "closing_total": pred.closing_total,
            "clv_spread": pred.clv_spread,
            "clv_total": pred.clv_total,
        },
    }


@router.get("")
async def list_predictions(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Get latest predictions for all upcoming games."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Subquery: latest prediction per game
    result = await db.execute(
        select(Prediction)
        .join(Game)
        .where(Game.status == "NS")
        .order_by(Game.commence_time, Prediction.predicted_at.desc())
    )
    predictions = result.scalars().all()

    # Deduplicate to latest per game
    seen: set[int] = set()
    latest: list[Prediction] = []
    for pred in predictions:
        game_id = int(cast(Any, pred.game_id))
        if game_id not in seen:
            seen.add(game_id)
            latest.append(pred)

    output = []
    for pred in latest:
        game_result = await db.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.id == pred.game_id)
        )
        game = game_result.scalar_one_or_none()
        if game:
            output.append(_format_prediction(pred, game))

    return {"predictions": output, "count": len(output)}


# ── Static routes MUST come before /{game_id} to avoid 422 ────────


@router.get("/slate.csv")
async def download_slate_csv(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Download the full daily slate as a CSV file."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    result = await db.execute(
        select(Prediction)
        .join(Game)
        .where(Game.status == "NS")
        .order_by(Game.commence_time, Prediction.predicted_at.desc())
    )
    predictions = result.scalars().all()

    seen: set[int] = set()
    latest: list[Prediction] = []
    for pred in predictions:
        game_id = int(cast(Any, pred.game_id))
        if game_id not in seen:
            seen.add(game_id)
            latest.append(pred)

    game_ids = [int(cast(Any, p.game_id)) for p in latest]
    game_result = await db.execute(
        select(Game)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
        )
        .where(Game.id.in_(game_ids))
        .order_by(Game.commence_time)
    )
    games = game_result.scalars().all()
    game_by_id = {int(cast(Any, g.id)): g for g in games}

    rows = []
    for pred in latest:
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game:
            rows.append((pred, game))

    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    csv_content = build_slate_csv(rows)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="nba_slate_{today}.csv"'
        },
    )


@router.get("/slate.html")
async def download_slate_html(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Serve the full daily slate as an interactive HTML page."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    from sqlalchemy import func as sa_func

    result = await db.execute(
        select(Prediction)
        .join(Game)
        .where(Game.status == "NS")
        .order_by(Game.commence_time, Prediction.predicted_at.desc())
    )
    predictions = result.scalars().all()

    seen: set[int] = set()
    latest: list[Prediction] = []
    for pred in predictions:
        game_id = int(cast(Any, pred.game_id))
        if game_id not in seen:
            seen.add(game_id)
            latest.append(pred)

    game_ids = [int(cast(Any, p.game_id)) for p in latest]
    game_result = await db.execute(
        select(Game)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
        )
        .where(Game.id.in_(game_ids))
        .order_by(Game.commence_time)
    )
    games = game_result.scalars().all()
    game_by_id = {int(cast(Any, g.id)): g for g in games}

    rows = []
    for pred in latest:
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game:
            rows.append((pred, game))

    odds_ts_result = await db.execute(
        select(sa_func.max(OddsSnapshot.captured_at))
    )
    odds_pulled_at = odds_ts_result.scalar_one_or_none()

    slate_body = build_html_slate(rows, odds_pulled_at=odds_pulled_at)
    csv_url = f"{settings.api_base_url}/predictions/slate.csv" if settings.api_base_url else "/predictions/slate.csv"
    html_page = (
        "<!DOCTYPE html>"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>NBA Daily Slate | GBSV</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "margin:0;padding:16px;background:#f5f5f5;color:#1a2332}"
        ".container{max-width:1200px;margin:0 auto;background:#fff;"
        "border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}"
        "table{font-size:13px}"
        "@media(max-width:768px){.container{padding:10px;border-radius:0}"
        "table{font-size:11px}th,td{padding:4px 3px!important}}"
        ".dl-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}"
        ".dl-btn{padding:8px 18px;border:none;border-radius:6px;font-weight:600;"
        "font-size:13px;cursor:pointer;text-decoration:none;display:inline-block}"
        ".dl-btn-primary{background:#1a2332;color:#d4af37}"
        ".dl-btn-secondary{background:#e9ecef;color:#1a2332}"
        "</style></head><body>"
        '<div class="container">'
        '<div class="dl-bar">'
        f'<a class="dl-btn dl-btn-secondary" href="{csv_url}">&#x1f4e5; Download CSV</a>'
        "</div>"
        f"{slate_body}"
        "</div></body></html>"
    )
    return Response(content=html_page, media_type="text/html")


@router.post("/publish/teams")
async def publish_predictions_to_teams(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Generate latest predictions and send formatted payload to Teams."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if (
        not (settings.teams_team_id and settings.teams_channel_id)
        and not settings.teams_webhook_url
    ):
        raise HTTPException(status_code=400, detail="Teams delivery is not configured")

    predictions = await predictor.predict_upcoming(db)
    if not predictions:
        return {"status": "ok", "published": 0, "detail": "No upcoming games"}

    game_ids = [int(cast(Any, p.game_id)) for p in predictions]
    game_result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.id.in_(game_ids))
        .order_by(Game.commence_time)
    )
    games = game_result.scalars().all()
    game_by_id = {int(cast(Any, g.id)): g for g in games}

    rows: list[tuple[Prediction, Game]] = []
    for pred in predictions:
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game is not None:
            rows.append((pred, game))

    # Latest odds pull timestamp
    from sqlalchemy import func as sa_func

    odds_ts_result = await db.execute(select(sa_func.max(OddsSnapshot.captured_at)))
    odds_pulled_at = odds_ts_result.scalar_one_or_none()

    download_url = (
        f"{settings.api_base_url}/predictions/slate.html"
        if settings.api_base_url
        else None
    )

    payload = build_teams_card(
        rows,
        settings.teams_max_games_per_message,
        odds_pulled_at=odds_pulled_at,
        download_url=download_url,
    )
    if settings.teams_team_id and settings.teams_channel_id:
        await send_card_via_graph(
            settings.teams_team_id, settings.teams_channel_id, payload
        )
    else:
        await send_card_to_teams(settings.teams_webhook_url, payload)
    return {"status": "ok", "published": len(rows)}


# ── Dynamic route MUST come after all static routes ──────────────


@router.get("/{game_id}")
async def get_prediction(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    """Get prediction detail for a specific game."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    game_result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.id == game_id)
    )
    game = game_result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    pred_result = await db.execute(
        select(Prediction)
        .where(Prediction.game_id == game_id)
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
    )
    pred = pred_result.scalar_one_or_none()
    if not pred:
        raise HTTPException(
            status_code=404, detail="No prediction available for this game"
        )

    result = _format_prediction(pred, game)

    # Add latest odds for edge comparison
    odds_result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.game_id == game_id)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(50)
    )
    odds = odds_result.scalars().all()
    if odds:
        import numpy as np

        spreads = [
            float(cast(Any, o.point))
            for o in odds
            if cast(Any, o.market) == "spreads" and o.point is not None
        ]
        totals = [
            float(cast(Any, o.point))
            for o in odds
            if cast(Any, o.market) == "totals" and o.point is not None
        ]

        if spreads:
            mkt_spread = float(np.mean(spreads))
            result["markets"]["fg_spread"]["market_line"] = mkt_spread
            result["markets"]["fg_spread"]["edge"] = round(
                _as_float(pred.fg_spread) - mkt_spread, 1
            )
        if totals:
            mkt_total = float(np.mean(totals))
            result["markets"]["fg_total"]["market_line"] = mkt_total
            result["markets"]["fg_total"]["edge"] = round(
                _as_float(pred.fg_total) - mkt_total, 1
            )

    return result
