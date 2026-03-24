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
    assert "runtime_status" in body
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


@pytest.mark.anyio
async def test_predictions_requires_ready_models_with_runtime_status_detail():
    class _DummyPredictor:
        is_ready = False

        def get_runtime_status(self):
            return {
                "ready": False,
                "reason": "Model artifact feature shape mismatch",
                "expected_features": 121,
            }

    app.dependency_overrides[get_predictor] = lambda: _DummyPredictor()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/predictions")
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["message"] == "Model artifact feature shape mismatch"
        assert detail["runtime_status"]["expected_features"] == 121
    finally:
        app.dependency_overrides.pop(get_predictor, None)
