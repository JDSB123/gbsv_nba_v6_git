"""Tests for odds routes and repositories (odds + models)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.db.repositories.models import ModelRepository
from src.db.repositories.odds import OddsRepository


# ── OddsRepository ────────────────────────────────────────────────


class TestOddsRepository:
    @pytest.mark.anyio
    async def test_get_latest_odds(self):
        session = AsyncMock()
        snap = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [snap]
        session.execute = AsyncMock(return_value=mock_result)
        repo = OddsRepository(session)
        result = await repo.get_latest_odds_for_upcoming_games()
        assert len(result) == 1


# ── ModelRepository ──────────────────────────────────────────────


class TestModelRepository:
    @pytest.mark.anyio
    async def test_get_all_models(self):
        session = AsyncMock()
        m = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [m]
        session.execute = AsyncMock(return_value=mock_result)
        repo = ModelRepository(session)
        result = await repo.get_all_models_ordered_by_creation()
        assert len(result) == 1

    @pytest.mark.anyio
    async def test_get_model_by_version(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock(model_version="v1")
        session.execute = AsyncMock(return_value=mock_result)
        repo = ModelRepository(session)
        result = await repo.get_model_by_version("v1")
        assert result is not None

    @pytest.mark.anyio
    async def test_get_finished_game_predictions(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(MagicMock(), MagicMock())]
        session.execute = AsyncMock(return_value=mock_result)
        repo = ModelRepository(session)
        result = await repo.get_finished_game_predictions()
        assert len(result) == 1


# ── /odds/latest route ───────────────────────────────────────────


@pytest.mark.anyio
async def test_odds_latest():
    snap = MagicMock(
        game_id=1,
        bookmaker="fanduel",
        market="spreads",
        outcome_name="Celtics",
        price=-110,
        point=-3.5,
        captured_at=datetime(2024, 3, 15, 12, 0, tzinfo=UTC),
    )
    with patch("src.api.routes.odds.async_session_factory") as mock_sf:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [snap]
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/odds/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["odds"][0]["bookmaker"] == "fanduel"
