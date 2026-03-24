from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_predictor
from src.api.rate_limit import limiter
from src.config import get_settings
from src.db.repositories.predictions import PredictionRepository
from src.db.session import get_db
from src.models.predictor import Predictor
from src.services.predictions import PredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _not_ready_detail(reason: str) -> dict[str, str]:
    return {"message": "Predictions not ready", "reason": reason}


async def get_prediction_service(
    db: AsyncSession = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
) -> PredictionService:
    repo = PredictionRepository(db)
    settings = get_settings()
    return PredictionService(repo, predictor, settings)


@router.get("")
@router.get("/")
@limiter.limit("60/minute")
async def list_predictions(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, Any]:
    if not service.predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    return await service.get_list_predictions()


@router.get("/slate.html", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def slate_html(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> HTMLResponse:
    """Return a filterable/sortable HTML slate of today's predictions."""
    if not service.predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    rows, odds_pulled_at = await service.get_slate_payload()
    if not rows:
        raise HTTPException(status_code=400, detail=_not_ready_detail("No predictions available"))

    from src.notifications.teams import build_html_slate

    html = build_html_slate(rows, odds_pulled_at=odds_pulled_at)
    return HTMLResponse(content=html)


@router.get("/slate.csv")
@limiter.limit("60/minute")
async def slate_csv(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> Response:
    """Return today's predictions as a downloadable CSV."""
    if not service.predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    rows, odds_pulled_at = await service.get_slate_payload()
    if not rows:
        raise HTTPException(status_code=400, detail=_not_ready_detail("No predictions available"))

    from src.notifications.teams import build_slate_csv

    csv_bytes = build_slate_csv(rows)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=slate.csv"},
    )


@router.get("/{game_id}")
@limiter.limit("60/minute")
async def get_prediction(
    request: Request,
    game_id: int,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, Any]:
    if not service.predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    detail = await service.get_prediction_detail(game_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if detail.get("pred") is None:
        raise HTTPException(
            status_code=400,
            detail=_not_ready_detail(f"No prediction found for game {game_id}"),
        )
    return detail["result"]


@router.post("/publish/teams")
@limiter.limit("10/minute")
async def publish_slate_to_teams(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, Any]:
    if not service.predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    settings = get_settings()
    if not settings.teams_webhook_url:
        raise HTTPException(status_code=500, detail="MSTEAMS_WEBHOOK_URL not configured")

    rows, odds_pulled_at = await service.get_slate_payload()
    if not rows:
        raise HTTPException(status_code=400, detail=_not_ready_detail("No predictions available"))

    from src.notifications.teams import build_teams_card, send_card_to_teams

    card = build_teams_card(
        rows,
        settings.teams_max_games_per_message,
        odds_pulled_at=odds_pulled_at,
    )
    await send_card_to_teams(settings.teams_webhook_url, card)

    return {"message": "Published slate to Teams successfully"}


@router.post("/publish/graph")
@limiter.limit("10/minute")
async def publish_slate_via_graph(
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, Any]:
    if not service.predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")

    settings = get_settings()
    if not settings.teams_team_id or not settings.teams_channel_id:
        raise HTTPException(
            status_code=500,
            detail="TEAMS_TEAM_ID and TEAMS_CHANNEL_ID not configured",
        )

    rows, odds_pulled_at = await service.get_slate_payload()
    if not rows:
        raise HTTPException(status_code=400, detail=_not_ready_detail("No predictions available"))

    from src.notifications.teams import build_teams_card, send_card_via_graph

    card = build_teams_card(
        rows,
        settings.teams_max_games_per_message,
        odds_pulled_at=odds_pulled_at,
    )
    await send_card_via_graph(settings.teams_team_id, settings.teams_channel_id, card)

    return {"message": "Published slate to Teams (Graph API) successfully"}


@router.post("/refresh")
@limiter.limit("5/minute")
async def refresh_predictions(
    request: Request,
) -> dict[str, Any]:
    """Pull fresh odds, generate new predictions, and publish to Teams.

    One-tap endpoint designed for mobile use — triggers the full pipeline:
    1. Fetch latest full-game odds from The Odds API
    2. Generate predictions for all upcoming games
    3. Publish formatted slate to Teams (Graph API or webhook)
    """
    from src.data.scheduler import generate_predictions_and_publish

    await generate_predictions_and_publish()
    return {"message": "Fresh odds pulled, predictions generated, and slate published"}
