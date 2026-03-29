"""Tests for PredictionRepository query methods."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.repositories.predictions import PredictionRepository


@pytest.fixture
def repo():
    mock_session = AsyncMock()
    return PredictionRepository(mock_session), mock_session


class TestGetLatestPredictionsForUpcomingGames:
    @pytest.mark.anyio
    async def test_returns_latest_per_game(self, repo):
        r, session = repo
        pred1 = MagicMock(game_id=1)
        pred2 = MagicMock(game_id=2)
        pred1_dup = MagicMock(game_id=1)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [pred1, pred2, pred1_dup]
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_predictions_for_upcoming_games()
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_empty(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_predictions_for_upcoming_games()
        assert result == []


class TestGetGamesWithTeamsAndStats:
    @pytest.mark.anyio
    async def test_returns_games(self, repo):
        r, session = repo
        game1 = MagicMock(id=1)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [game1]
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_games_with_teams_and_stats([1])
        assert len(result) == 1


class TestGetGamesWithTeams:
    @pytest.mark.anyio
    async def test_returns_games(self, repo):
        r, session = repo
        game1 = MagicMock(id=1)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [game1]
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_games_with_teams([1])
        assert len(result) == 1


class TestGetGameWithTeams:
    @pytest.mark.anyio
    async def test_found(self, repo):
        r, session = repo
        game = MagicMock(id=42)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = game
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_game_with_teams(42)
        assert result is not None

    @pytest.mark.anyio
    async def test_not_found(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_game_with_teams(999)
        assert result is None


class TestGetLatestPredictionForGame:
    @pytest.mark.anyio
    async def test_found(self, repo):
        r, session = repo
        pred = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = pred
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_prediction_for_game(1)
        assert result is pred

    @pytest.mark.anyio
    async def test_not_found(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_prediction_for_game(999)
        assert result is None


class TestGetRecentOddsSnapshots:
    @pytest.mark.anyio
    async def test_returns_snapshots(self, repo):
        r, session = repo
        snap = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [snap]
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_recent_odds_snapshots(1)
        assert len(result) == 1


class TestGetLatestOddsPullTimestamp:
    @pytest.mark.anyio
    async def test_returns_timestamp(self, repo):
        r, session = repo
        from datetime import datetime, UTC
        ts = datetime.now(UTC)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = ts
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_odds_pull_timestamp()
        assert result == ts

    @pytest.mark.anyio
    async def test_no_data(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_odds_pull_timestamp()
        assert result is None
