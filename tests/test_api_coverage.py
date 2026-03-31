"""Tests targeting health.py deep-check error/warning paths,
api/main.py API key auth rejection + lifespan, and rate_limit.py edge cases."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_predictor
from src.api.main import app
from src.db.session import get_db


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_api_lifespan_runs_startup_and_shutdown():
    from src.api.main import lifespan

    with (
        patch("src.api.main._setup_logging") as mock_setup,
        patch("src.api.main.get_settings") as mock_settings,
        patch("src.api.main.logger") as mock_logger,
    ):
        mock_settings.return_value.app_env = "test"
        async with lifespan(app):
            pass

    mock_setup.assert_called_once()
    assert mock_logger.info.call_count >= 3


@pytest.mark.anyio
async def test_get_db_yields_session_once():
    mock_session = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.session.async_session_factory", return_value=mock_ctx):
        gen = get_db()
        yielded = await anext(gen)
        assert yielded is mock_session
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


# ── Health deep: database error (lines 34-36) ────────────────────
@pytest.mark.anyio
async def test_health_deep_database_error():
    """When the DB throws, deep health should report degraded."""

    async def _broken_db():
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=Exception("Connection refused"))
        yield db

    class _ReadyPredictor:
        is_ready = True

        def get_runtime_status(self):
            return {"model_version": "v6-test", "compatibility_mode": False}

    app.dependency_overrides[get_db] = _broken_db
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["database"]["status"] == "error"
    finally:
        app.dependency_overrides.clear()


# ── Health deep: no odds data warning (lines 65-66) ──────────────
@pytest.mark.anyio
async def test_health_deep_no_odds_data():
    """When odds query returns None, health should warn."""
    call_count = 0

    async def _db_factory():
        db = AsyncMock()
        nonlocal call_count

        async def _exec(stmt, *a, **kw):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # "SELECT 1"
                result.scalar_one_or_none = MagicMock(return_value=1)
            elif call_count == 2:
                # max(OddsSnapshot.captured_at) → None
                result.scalar_one_or_none = MagicMock(return_value=None)
            else:
                # team stats count → 30
                result.scalar = MagicMock(return_value=30)
            return result

        db.execute = AsyncMock(side_effect=_exec)
        yield db

    class _ReadyPredictor:
        is_ready = True

        def get_runtime_status(self):
            return {"model_version": "v6-test", "compatibility_mode": False}

    app.dependency_overrides[get_db] = _db_factory
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        data = resp.json()
        assert data["checks"]["odds_freshness"]["status"] == "warning"
        assert "No odds data" in data["checks"]["odds_freshness"]["detail"]
    finally:
        app.dependency_overrides.clear()


# ── Health deep: team stats exception (lines 77-78) ──────────────
@pytest.mark.anyio
async def test_health_deep_team_stats_error():
    """When team stats query throws, health should catch it."""
    call_count = 0

    async def _db_factory():
        db = AsyncMock()
        nonlocal call_count

        async def _exec(stmt, *a, **kw):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none = MagicMock(return_value=1)
            elif call_count == 2:
                result.scalar_one_or_none = MagicMock(return_value=None)
            else:
                raise Exception("team stats table error")
            return result

        db.execute = AsyncMock(side_effect=_exec)
        yield db

    class _ReadyPredictor:
        is_ready = True

        def get_runtime_status(self):
            return {"model_version": "v6-test", "compatibility_mode": False}

    app.dependency_overrides[get_db] = _db_factory
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        data = resp.json()
        assert data["checks"]["team_stats"]["status"] == "error"
    finally:
        app.dependency_overrides.clear()


# ── API key auth rejection (lines 76-80) ──────────────────────────
@pytest.mark.anyio
async def test_api_key_auth_rejects_wrong_key():
    """When API key is configured but wrong key is sent → 401."""
    with patch("src.api.main.get_settings") as mock_settings:
        mock_settings.return_value.api_key = "correct-key-123"
        mock_settings.return_value.api_base_url = ""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/predictions",
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]


@pytest.mark.anyio
async def test_api_key_auth_allows_correct_key():
    """When correct API key is provided, request proceeds."""
    from src.api.dependencies import get_predictor
    from src.db.session import get_db

    class _DummyPredictor:
        is_ready = False

        def get_runtime_status(self):
            return {"ready": False}

    app.dependency_overrides[get_predictor] = lambda: _DummyPredictor()
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    try:
        with patch("src.api.main.get_settings") as mock_settings:
            mock_settings.return_value.api_key = "correct-key-123"
            mock_settings.return_value.api_base_url = ""
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/predictions",
                    headers={"X-API-Key": "correct-key-123"},
                )
            # Should pass auth but fail on predictor not ready (503)
            assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_root_is_public_when_api_key_is_configured():
    with patch("src.api.main.get_settings") as mock_settings:
        mock_settings.return_value.api_key = "correct-key-123"
        mock_settings.return_value.api_base_url = ""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "Live API is running" in resp.text


# ── rate_limit.py: _read_file_utf8 FileNotFoundError (lines 18-19) ──
def test_rate_limit_read_file_utf8_missing_file():
    from src.api.rate_limit import _read_file_utf8

    result = _read_file_utf8(None, "/nonexistent/path/.env")
    assert result == {}


# ── model route: /model/performance (line 55) ────────────────────
@pytest.mark.anyio
async def test_model_performance_returns_data():
    from src.api.routes.model import get_model_service

    mock_service = AsyncMock()
    mock_service.get_performance = AsyncMock(return_value={"models": {}})
    app.dependency_overrides[get_model_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model/performance")
        assert resp.status_code == 200
        assert resp.json() == {"models": {}}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_get_model_service_wraps_repository_session():
    from src.api.routes.model import get_model_service

    db = AsyncMock()
    service = await get_model_service(db)

    assert service.repo._session is db


# ── predictions routes: no predictions 400 (line 73) ─────────────
@pytest.mark.anyio
async def test_predictions_slate_csv_no_predictions():
    from src.api.routes.predictions import get_prediction_service

    mock_service = MagicMock()
    mock_service.predictor = SimpleNamespace(is_ready=True)
    mock_service.get_slate_payload = AsyncMock(return_value=([], None))

    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/slate.csv")
        assert resp.status_code == 200
        assert resp.headers["x-slate-status"] == "empty"
    finally:
        app.dependency_overrides.clear()


# ── predictions routes: detail result (line 102) ─────────────────
@pytest.mark.anyio
async def test_prediction_detail_returns_result():
    from src.api.routes.predictions import get_prediction_service

    mock_service = MagicMock()
    mock_service.predictor = SimpleNamespace(is_ready=True)
    mock_service.get_prediction_detail = AsyncMock(
        return_value={
            "pred": {"game_id": 1},
            "result": {"game_id": 1, "home": "Lakers", "away": "Celtics"},
        }
    )

    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/1")
        assert resp.status_code == 200
        assert resp.json()["game_id"] == 1
    finally:
        app.dependency_overrides.clear()


# ── predictions routes: detail no pred → 400 ─────────────────────
@pytest.mark.anyio
async def test_prediction_detail_no_pred_400():
    from src.api.routes.predictions import get_prediction_service

    mock_service = MagicMock()
    mock_service.predictor = SimpleNamespace(is_ready=True)
    mock_service.get_prediction_detail = AsyncMock(
        return_value={"game": {}, "pred": None}
    )

    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/999")
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()
