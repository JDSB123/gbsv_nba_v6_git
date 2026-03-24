from datetime import UTC, datetime
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_predictor
from src.config import get_settings
from src.db.models import Game, Prediction
from src.db.session import get_db
from src.db.repositories.predictions import PredictionRepository
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


def _not_ready_detail(predictor: Predictor) -> Any:
    # Return a detailed readiness payload when available.
    status_fn = getattr(predictor, "get_runtime_status", None)
    if callable(status_fn):
        status = status_fn()
        reason = status.get("reason")
        if reason:
            return {"message": reason, "runtime_status": status}
        return {"message": "Models not loaded", "runtime_status": status}
    return "Models not loaded"


def _parse_iso_utc(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _odds_freshness_summary(
    predictions: list[dict],
    max_age_minutes: int,
) -> dict[str, Any]:
    now_utc = datetime.now(UTC)
    ages_minutes: list[float] = []
    missing_odds_sourced = 0
    missing_captured_at = 0

    for row in predictions:
        odds_sourced = row.get("odds_sourced")
        if not isinstance(odds_sourced, dict):
            missing_odds_sourced += 1
            continue

        captured_at = _parse_iso_utc(odds_sourced.get("captured_at"))
        if captured_at is None:
            missing_captured_at += 1
            continue

        ages_minutes.append((now_utc - captured_at).total_seconds() / 60.0)

    stale_count = sum(1 for age in ages_minutes if age > max_age_minutes)
    usable = len(ages_minutes)
    status = "ok"
    if missing_odds_sourced or missing_captured_at or stale_count:
        status = "warning"

    return {
        "status": status,
        "max_allowed_age_minutes": max_age_minutes,
        "evaluated_predictions": len(predictions),
        "usable_captured_at_count": usable,
        "missing_odds_sourced": missing_odds_sourced,
        "missing_captured_at": missing_captured_at,
        "stale_count": stale_count,
        "freshest_age_minutes": round(min(ages_minutes), 2) if ages_minutes else None,
        "stale_threshold_minutes": max_age_minutes,
        "stale_ratio": round(stale_count / usable, 3) if usable else None,
    }


def _format_prediction(pred: Prediction, game: Game) -> dict:
    home_name = game.home_team.name if game.home_team is not None else f"Team {game.home_team_id}"
    away_name = game.away_team.name if game.away_team is not None else f"Team {game.away_team_id}"
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
        "predicted_at": (pred.predicted_at.isoformat() if pred.predicted_at is not None else None),
        "odds_sourced": pred.odds_sourced,
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
    # Get latest predictions for all upcoming games.
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=_not_ready_detail(predictor))

    repo = PredictionRepository(db)
    latest = await repo.get_latest_predictions_for_upcoming_games()

    game_ids = [int(cast(Any, p.game_id)) for p in latest if p.game_id is not None]
    games = await repo.get_games_with_teams(game_ids)
    game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

    output = []
    for pred in latest:
        if pred.game_id is None:
            continue
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game:
            output.append(_format_prediction(pred, game))

    freshness = _odds_freshness_summary(output, settings.odds_freshness_max_age_minutes)
    return {"predictions": output, "count": len(output), "freshness": freshness}


# == Static routes MUST come before /{game_id} to avoid 422 ==


@router.get("/slate.csv")
async def download_slate_csv(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    # Download the full daily slate as a CSV file.
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=_not_ready_detail(predictor))

    repo = PredictionRepository(db)
    latest = await repo.get_latest_predictions_for_upcoming_games()

    game_ids = [int(cast(Any, p.game_id)) for p in latest if p.game_id is not None]
    games = await repo.get_games_with_teams_and_stats(game_ids)
    game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

    rows = []
    for pred in latest:
        if pred.game_id is None:
            continue
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game:
            rows.append((pred, game))

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    csv_content = build_slate_csv(rows)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="nba_slate_{today}.csv"'},
    )


@router.get("/slate.html")
async def download_slate_html(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=_not_ready_detail(predictor))

    repo = PredictionRepository(db)
    latest = await repo.get_latest_predictions_for_upcoming_games()

    game_ids = [int(cast(Any, p.game_id)) for p in latest if p.game_id is not None]
    games = await repo.get_games_with_teams_and_stats(game_ids)
    game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

    rows = []
    for pred in latest:
        if pred.game_id is None:
            continue
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game:
            rows.append((pred, game))

    odds_pulled_at = await repo.get_latest_odds_pull_timestamp()

    slate_body = build_html_slate(rows, odds_pulled_at=odds_pulled_at)
    csv_url = (
        f"{settings.api_base_url}/predictions/slate.csv"
        if settings.api_base_url
        else "/predictions/slate.csv"
    )

    html_page = (
        "<!DOCTYPE html>"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>NBA Daily Slate | GBSV</title>"
        "<style>"
        'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:16px;background:#f5f5f5;color:#1a2332}'
        ".container{max-width:1200px;margin:0 auto;background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}"
        "table{font-size:13px}"
        "@media(max-width:768px){.container{padding:10px;border-radius:0}table{font-size:11px}th,td{padding:4px 3px!important}}"
        ".dl-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}"
        ".dl-btn{padding:8px 18px;border:none;border-radius:6px;font-weight:600;font-size:13px;cursor:pointer;text-decoration:none;display:inline-block}"
        ".dl-btn-primary{background:#1a2332;color:#d4af37}"
        ".dl-btn-secondary{background:#e9ecef;color:#1a2332}"
        "</style></head><body>"
        '<div class="container">'
        '<div class="dl-bar">'
        f'<a class="dl-btn dl-btn-secondary" href="{csv_url}">Download CSV</a>'
        '<a class="dl-btn dl-btn-secondary" href="/performance/dashboard">Performance</a>'
        "</div>"
        f"{slate_body}"
        "</div></body></html>"
    )

    return Response(content=html_page, media_type="text/html")


@router.get("/refresh")
async def refresh_predictions(
    force_odds_pull: bool = True,
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=_not_ready_detail(predictor))

    odds_pull_warning: str | None = None
    if force_odds_pull:
        from src.data.scheduler import poll_fg_odds

        try:
            await poll_fg_odds()
        except Exception as exc:
            odds_pull_warning = str(exc)

    try:
        predictions = await predictor.predict_upcoming(db)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Prediction refresh failed",
                "error": str(exc),
                "runtime_status": predictor.get_runtime_status(),
            },
        ) from exc

    if not predictions:
        return {
            "status": "ok",
            "refreshed": 0,
            "detail": "No upcoming games",
            "force_odds_pull": force_odds_pull,
            "odds_pull_warning": odds_pull_warning,
        }

    repo = PredictionRepository(db)
    game_ids = [int(cast(Any, p.game_id)) for p in predictions if p.game_id is not None]
    games = await repo.get_games_with_teams_and_stats(game_ids)
    game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

    output = []
    for pred in predictions:
        if pred.game_id is None:
            continue
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game:
            output.append(_format_prediction(pred, game))

    freshness = _odds_freshness_summary(output, settings.odds_freshness_max_age_minutes)
    return {
        "status": "ok",
        "refreshed": len(output),
        "model_version": predictions[0].model_version if predictions else None,
        "force_odds_pull": force_odds_pull,
        "odds_pull_warning": odds_pull_warning,
        "freshness": freshness,
        "predictions": output,
    }


@router.post("/publish/teams")
async def publish_predictions_to_teams(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=_not_ready_detail(predictor))

    if (
        not (settings.teams_team_id and settings.teams_channel_id)
        and not settings.teams_webhook_url
    ):
        raise HTTPException(status_code=400, detail="Teams delivery is not configured")

    predictions = await predictor.predict_upcoming(db)
    if not predictions:
        return {"status": "ok", "published": 0, "detail": "No upcoming games"}

    repo = PredictionRepository(db)
    game_ids = [int(cast(Any, p.game_id)) for p in predictions if p.game_id is not None]
    games = await repo.get_games_with_teams_and_stats(game_ids)
    game_by_id = {int(cast(Any, g.id)): g for g in games if g.id is not None}

    rows: list[tuple[Prediction, Game]] = []
    for pred in predictions:
        if pred.game_id is None:
            continue
        game = game_by_id.get(int(cast(Any, pred.game_id)))
        if game is not None:
            rows.append((pred, game))

    odds_pulled_at = await repo.get_latest_odds_pull_timestamp()

    download_url = (
        f"{settings.api_base_url}/predictions/slate.html" if settings.api_base_url else None
    )

    payload = build_teams_card(
        rows,
        settings.teams_max_games_per_message,
        odds_pulled_at=odds_pulled_at,
        download_url=download_url,
    )
    if settings.teams_team_id and settings.teams_channel_id:
        await send_card_via_graph(settings.teams_team_id, settings.teams_channel_id, payload)
    else:
        await send_card_to_teams(settings.teams_webhook_url, payload)
    return {"status": "ok", "published": len(rows)}


@router.get("/{game_id}")
async def get_prediction(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=_not_ready_detail(predictor))

    repo = PredictionRepository(db)
    game = await repo.get_game_with_teams(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    pred = await repo.get_latest_prediction_for_game(game_id)
    if not pred:
        raise HTTPException(status_code=404, detail="No prediction available for this game")

    result = _format_prediction(pred, game)

    # Add latest odds for edge comparison
    odds = await repo.get_recent_odds_snapshots(game_id, limit=50)
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
            result["markets"]["fg_total"]["edge"] = round(_as_float(pred.fg_total) - mkt_total, 1)

    return result
