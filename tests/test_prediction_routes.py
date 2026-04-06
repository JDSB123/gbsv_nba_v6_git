"""Tests for prediction and model API routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_predictor
from src.api.main import app
from src.db.session import get_db


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _ReadyPredictor:
    is_ready = True
    model_version = "v6.0"

    def get_runtime_status(self):
        return {"model_version": "v6.0", "compatibility_mode": False}

    def get_metrics(self):
        return {"spread_rmse": 8.5}

    def get_feature_importance(self):
        return {"elo": 0.15}


class _NotReadyPredictor:
    is_ready = False
    model_version = None

    def get_runtime_status(self):
        return {}

    def get_metrics(self):
        return {}

    def get_feature_importance(self):
        return {}


# ── _not_ready_detail ────────────────────────────────────────────


def test_not_ready_detail():
    from src.api.routes.predictions import _not_ready_detail

    result = _not_ready_detail("test reason")
    assert result["message"] == "Predictions not ready"
    assert result["reason"] == "test reason"


# ── Prediction endpoints ────────────────────────────────────────


@pytest.mark.anyio
async def test_list_predictions_not_ready():
    app.dependency_overrides[get_predictor] = lambda: _NotReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_slate_html_not_ready():
    app.dependency_overrides[get_predictor] = lambda: _NotReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/slate.html")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_slate_csv_not_ready():
    app.dependency_overrides[get_predictor] = lambda: _NotReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/slate.csv")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_teams_not_ready():
    app.dependency_overrides[get_predictor] = lambda: _NotReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predictions/publish/teams")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_graph_not_ready():
    app.dependency_overrides[get_predictor] = lambda: _NotReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predictions/publish/graph")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_teams_no_webhook():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.api.routes.predictions.get_settings") as mock_s:
            mock_s.return_value = MagicMock(teams_webhook_url=None)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/predictions/publish/teams")
            assert resp.status_code == 500
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_graph_no_config():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.api.routes.predictions.get_settings") as mock_s:
            mock_s.return_value = MagicMock(teams_team_id=None, teams_channel_id=None)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/predictions/publish/graph")
            assert resp.status_code == 500
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_get_prediction_not_found():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.services.predictions.PredictionService.get_prediction_detail", new_callable=AsyncMock, return_value=None):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions/99999")
            assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ── /model endpoints ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_model_status():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["active_model_version"] == "v6.0"
        assert "metrics" in data
        assert "feature_importance" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_model_registry():
    mock_db = AsyncMock()
    mock_rows = [
        MagicMock(
            model_version="v6.0",
            is_active=True,
            promoted_at=None,
            retired_at=None,
            promotion_reason="initial",
            created_at=None,
        ),
    ]
    mock_db.execute = AsyncMock(
        return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=mock_rows))))
    )

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_model_retrain():
    with patch("src.db.session.async_session_factory") as mock_sf:
        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("src.models.trainer.ModelTrainer") as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer.train = AsyncMock(return_value={"rmse": 5.0})
            mock_trainer_cls.return_value = mock_trainer
            with patch("src.api.routes.model.get_settings") as mock_gs:
                mock_settings = MagicMock()
                mock_settings.api_key = "test-key"
                mock_gs.return_value = mock_settings
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/model/retrain", headers={"X-API-Key": "test-key"}
                    )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "complete"


@pytest.mark.anyio
async def test_model_retrain_requires_auth():
    """Retrain endpoint rejects requests without a valid API key."""
    with patch("src.api.routes.model.get_settings") as mock_gs:
        mock_settings = MagicMock()
        mock_settings.api_key = "real-secret"
        mock_gs.return_value = mock_settings
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model/retrain")
        assert resp.status_code == 401


@pytest.mark.anyio
async def test_model_retrain_rejects_unconfigured_key():
    """Retrain endpoint returns 403 if api_key is not configured on server."""
    with patch("src.api.routes.model.get_settings") as mock_gs:
        mock_settings = MagicMock()
        mock_settings.api_key = ""
        mock_gs.return_value = mock_settings
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model/retrain")
        assert resp.status_code == 403


@pytest.mark.anyio
async def test_refresh_predictions():
    with patch(
        "src.data.scheduler.generate_predictions_and_publish",
        new_callable=AsyncMock,
        return_value=3,
    ) as mock_gen:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predictions/refresh")
        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert resp.json()["count"] == 3


@pytest.mark.anyio
async def test_refresh_predictions_no_fresh_rows():
    with patch(
        "src.data.scheduler.generate_predictions_and_publish",
        new_callable=AsyncMock,
        return_value=0,
    ) as mock_gen:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predictions/refresh")
        assert resp.status_code == 400
        mock_gen.assert_called_once()


# ── Happy-path prediction routes ─────────────────────────────────


@pytest.mark.anyio
async def test_list_predictions_success():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "src.services.predictions.PredictionService.get_list_predictions",
            new_callable=AsyncMock,
            return_value={"predictions": [], "odds_freshness": {}},
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions")
            assert resp.status_code == 200
            assert "predictions" in resp.json()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_slate_html_success():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with (
            patch(
            "src.services.predictions.PredictionService.get_slate_payload",
            new_callable=AsyncMock,
            return_value=([("pred_mock", "game_mock")], None),
            ),
            patch(
                "src.notifications.teams.build_html_slate",
                return_value="<html>slate</html>",
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions/slate.html")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_slate_csv_success():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with (
            patch(
                "src.services.predictions.PredictionService.get_slate_payload",
                new_callable=AsyncMock,
                return_value=([("pred_mock", "game_mock")], None),
            ),
            patch(
                "src.notifications.teams.build_slate_csv",
                return_value=b"col1,col2\nval1,val2\n",
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions/slate.csv")
            assert resp.status_code == 200
            assert "text/csv" in resp.headers["content-type"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_slate_html_no_rows():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with (
            patch(
                "src.services.predictions.PredictionService.get_slate_payload",
                new_callable=AsyncMock,
                return_value=([], None),
            ),
            patch(
                "src.services.predictions.PredictionService.get_list_predictions",
                new_callable=AsyncMock,
                return_value={
                    "predictions": [],
                    "freshness": {
                        "evaluated_predictions": 3,
                        "stale_count": 3,
                        "filtered_out_non_fresh": 3,
                        "freshest_age_minutes": 42.5,
                    },
                },
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions/slate.html")
            assert resp.status_code == 200
            assert resp.headers["x-slate-status"] == "empty"
            assert "No fresh predictions are available right now" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_slate_csv_no_rows_returns_header_only():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "src.services.predictions.PredictionService.get_slate_payload",
            new_callable=AsyncMock,
            return_value=([], None),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions/slate.csv")
            assert resp.status_code == 200
            assert resp.headers["x-slate-status"] == "empty"
            assert "Matchup" in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_get_prediction_no_pred():
    """Game exists but no prediction → 400."""
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "src.services.predictions.PredictionService.get_prediction_detail",
            new_callable=AsyncMock,
            return_value={"pred": None, "result": {}},
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/predictions/12345")
            assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_teams_success():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.api.routes.predictions.get_settings") as mock_s:
            mock_s.return_value = MagicMock(
                teams_webhook_url="https://hook.example.com",
                api_base_url="https://api.example.com",
                teams_max_games_per_message=10,
            )
            with (
                patch(
                    "src.services.predictions.PredictionService.get_slate_payload",
                    new_callable=AsyncMock,
                    return_value=([("pred", "game")], None),
                ),
                patch(
                    "src.notifications.teams.build_teams_card",
                    return_value={"card": "payload"},
                ),
                patch(
                    "src.notifications.teams.send_card_to_teams",
                    new_callable=AsyncMock,
                ) as mock_send,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/predictions/publish/teams")
                assert resp.status_code == 200
                mock_send.assert_called_once()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_teams_no_rows():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.api.routes.predictions.get_settings") as mock_s:
            mock_s.return_value = MagicMock(teams_webhook_url="https://hook.example.com")
            with patch(
                "src.services.predictions.PredictionService.get_slate_payload",
                new_callable=AsyncMock,
                return_value=([], None),
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/predictions/publish/teams")
                assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_graph_success():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.api.routes.predictions.get_settings") as mock_s:
            mock_s.return_value = MagicMock(
                teams_team_id="team-123",
                teams_channel_id="channel-456",
                teams_max_games_per_message=10,
            )
            with (
                patch(
                    "src.services.predictions.PredictionService.get_slate_payload",
                    new_callable=AsyncMock,
                    return_value=([("pred", "game")], None),
                ),
                patch(
                    "src.notifications.teams.build_html_slate",
                    return_value="<html>slate</html>",
                ),
                patch(
                    "src.notifications.teams.send_html_via_graph",
                    new_callable=AsyncMock,
                ) as mock_send,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/predictions/publish/graph")
                assert resp.status_code == 200
                mock_send.assert_called_once()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_publish_graph_no_rows():
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    mock_db = AsyncMock()

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        with patch("src.api.routes.predictions.get_settings") as mock_s:
            mock_s.return_value = MagicMock(
                teams_team_id="team-123",
                teams_channel_id="channel-456",
            )
            with patch(
                "src.services.predictions.PredictionService.get_slate_payload",
                new_callable=AsyncMock,
                return_value=([], None),
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/predictions/publish/graph")
                assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()
