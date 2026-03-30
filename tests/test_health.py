"""Tests for health endpoints (deep health check, freshness dashboard)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_predictor
from src.api.main import app
from src.db.session import get_db


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _mock_db():
    """Build a mock async session that returns configurable scalars."""
    db = AsyncMock()
    return db


def _scalar_result(value):
    """Create a mock result object that returns a scalar value."""
    mock = MagicMock()
    mock.scalar_one_or_none.return_value = value
    mock.scalar.return_value = value
    return mock


# ── /health ──────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health_returns_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── /health/deep ─────────────────────────────────────────────


class _ReadyPredictor:
    is_ready = True

    def get_runtime_status(self):
        return {"model_version": "v1.0", "compatibility_mode": False}


class _NotReadyPredictor:
    is_ready = False

    def get_runtime_status(self):
        return {"model_version": None, "compatibility_mode": False}


@pytest.mark.anyio
async def test_health_deep_all_ok():
    mock_db = AsyncMock()

    # DB connectivity check
    recent_odds = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=10)

    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # text("SELECT 1")
            return MagicMock()
        elif call_count == 2:
            # max(OddsSnapshot.captured_at)
            return _scalar_result(recent_odds)
        elif call_count == 3:
            # count(TeamSeasonStats)
            return _scalar_result(30)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["checks"]["database"]["status"] == "ok"
        assert data["checks"]["models"]["status"] == "ok"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_deep_models_not_ready():
    mock_db = AsyncMock()

    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock()
        elif call_count == 2:
            return _scalar_result(datetime.now(UTC).replace(tzinfo=None))
        elif call_count == 3:
            return _scalar_result(30)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_predictor] = lambda: _NotReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["models"]["status"] == "error"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_deep_no_odds():
    mock_db = AsyncMock()
    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock()
        elif call_count == 2:
            return _scalar_result(None)
        elif call_count == 3:
            return _scalar_result(30)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["odds_freshness"]["status"] == "warning"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_deep_stale_odds():
    mock_db = AsyncMock()
    # Odds 2 hours old
    stale_odds = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)
    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock()
        elif call_count == 2:
            return _scalar_result(stale_odds)
        elif call_count == 3:
            return _scalar_result(30)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_predictor] = lambda: _ReadyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/deep")
        data = resp.json()
        assert data["checks"]["odds_freshness"]["status"] == "warning"
        assert data["checks"]["odds_freshness"]["age_minutes"] > 60
    finally:
        app.dependency_overrides.clear()


# ── /health/freshness ────────────────────────────────────────


@pytest.mark.anyio
async def test_health_freshness_all_data():
    mock_db = AsyncMock()
    now = datetime.now(UTC).replace(tzinfo=None)

    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # max(OddsSnapshot.captured_at) - fresh
            return _scalar_result(now - timedelta(minutes=5))
        elif call_count == 2:
            # max(Injury.reported_at)
            return _scalar_result(now - timedelta(hours=1))
        elif call_count == 3:
            # max(Prediction.predicted_at)
            return _scalar_result(now - timedelta(minutes=30))
        elif call_count == 4:
            # count NS
            return _scalar_result(8)
        elif call_count == 5:
            # count FT
            return _scalar_result(100)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/freshness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["odds"]["status"] == "fresh"
        assert data["injuries"]["status"] == "fresh"
        assert "predictions" in data
        assert data["games"]["upcoming"] == 8
        assert data["games"]["completed"] == 100
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_freshness_missing_data():
    mock_db = AsyncMock()
    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count in (1, 2, 3):
            return _scalar_result(None)
        elif call_count == 4 or call_count == 5:
            return _scalar_result(0)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/freshness")
        data = resp.json()
        assert data["odds"]["status"] == "missing"
        assert data["injuries"]["status"] == "missing"
        assert data["predictions"]["status"] == "missing"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_freshness_stale_odds():
    mock_db = AsyncMock()
    now = datetime.now(UTC).replace(tzinfo=None)
    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Odds from 90 minutes ago - stale
            return _scalar_result(now - timedelta(minutes=90))
        elif call_count == 2 or call_count == 3:
            return _scalar_result(None)
        elif call_count == 4 or call_count == 5:
            return _scalar_result(0)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/freshness")
        data = resp.json()
        assert data["odds"]["status"] == "stale"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_health_freshness_very_stale_odds():
    mock_db = AsyncMock()
    now = datetime.now(UTC).replace(tzinfo=None)
    call_count = 0

    async def mock_execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _scalar_result(now - timedelta(hours=3))
        elif call_count == 2 or call_count == 3:
            return _scalar_result(None)
        elif call_count == 4 or call_count == 5:
            return _scalar_result(0)
        return _scalar_result(None)

    mock_db.execute = mock_execute

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/freshness")
        data = resp.json()
        assert data["odds"]["status"] == "very_stale"
    finally:
        app.dependency_overrides.clear()
