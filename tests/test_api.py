import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_predictor
from src.api.main import app
from src.models.predictor import MODEL_VERSION


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_model_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/model/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "ready" in body
    assert "version" in body
    assert body["version"] == MODEL_VERSION


@pytest.mark.anyio
async def test_predictions_requires_ready_models_list():
    class _DummyPredictor:
        is_ready = False

    app.dependency_overrides[get_predictor] = lambda: _DummyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Models not loaded"
    finally:
        app.dependency_overrides.pop(get_predictor, None)


@pytest.mark.anyio
async def test_predictions_requires_ready_models_detail():
    class _DummyPredictor:
        is_ready = False

    app.dependency_overrides[get_predictor] = lambda: _DummyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions/1")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Models not loaded"
    finally:
        app.dependency_overrides.pop(get_predictor, None)
