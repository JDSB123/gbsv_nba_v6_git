from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_predictor
from src.api.main import app
from src.config import Settings
from src.services.predictions import PredictionService


@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200

@pytest.mark.anyio
async def test_model_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/model/status")
    assert resp.status_code == 200

@pytest.mark.anyio
async def test_predictions_requires_ready_models_list():
    class _DummyPredictor:
        is_ready = False
        def get_runtime_status(self):
            return {"ready": False, "reason": "Not loaded"}
            
    app.dependency_overrides[get_predictor] = lambda: _DummyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Models not ready"
    finally:
        app.dependency_overrides.clear()

@pytest.mark.anyio
async def test_predictions_requires_ready_models_detail():
    class _DummyPredictor:
        is_ready = False
        def get_runtime_status(self):
            return {"ready": False, "reason": "Not loaded"}

    app.dependency_overrides[get_predictor] = lambda: _DummyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/1")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Models not ready"
    finally:
        app.dependency_overrides.clear()

def test_odds_freshness_summary_flags_stale_and_missing():
    service = PredictionService(None, None, Settings(odds_freshness_max_age_minutes=1))
    stale_time = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    rows = [
        {"odds_sourced": {"captured_at": stale_time}},
        {"odds_sourced": {"captured_at": None}},
        {"odds_sourced": None},
    ]

    summary = service.evaluate_odds_freshness(rows)
    assert summary["status"] == "warning"
    assert summary["missing_odds_sourced"] == 1
    assert summary["missing_captured_at"] == 1
    assert summary["stale_count"] == 1
