"""Tests for PredictionRepository query methods."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import sqlite

from src.db.repositories.predictions import PredictionRepository


@pytest.fixture
def repo():
    mock_session = AsyncMock()
    return PredictionRepository(mock_session), mock_session


class TestGetLatestPredictionsForUpcomingGames:
    @pytest.mark.anyio
    async def test_returns_latest_per_game(self, repo):
        r, session = repo
        pred1 = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC) - timedelta(minutes=10),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": "2026-03-30T05:10:00Z"},
        )
        pred2 = MagicMock(
            game_id=2,
            predicted_at=datetime.now(UTC) - timedelta(minutes=9),
            predicted_home_fg=108.0,
            predicted_away_fg=101.0,
            predicted_home_1h=54.0,
            predicted_away_1h=50.0,
            fg_spread=7.0,
            fg_total=209.0,
            h1_spread=4.0,
            h1_total=104.0,
            odds_sourced={"captured_at": "2026-03-30T05:11:00Z"},
        )
        pred1_dup = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC) - timedelta(minutes=5),
            predicted_home_fg=111.0,
            predicted_away_fg=104.0,
            predicted_home_1h=56.0,
            predicted_away_1h=51.0,
            fg_spread=7.0,
            fg_total=215.0,
            h1_spread=5.0,
            h1_total=107.0,
            odds_sourced={"captured_at": "2026-03-30T05:15:00Z"},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [pred1, pred2, pred1_dup]
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_predictions_for_upcoming_games()
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_prefers_odds_backed_prediction_over_newer_invalid_row(self, repo):
        r, session = repo
        good_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC) - timedelta(minutes=5),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": "2026-03-30T05:20:00Z"},
        )
        bad_newer_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": None},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [bad_newer_pred, good_pred]
        session.execute = AsyncMock(return_value=mock_result)

        result = await r.get_latest_predictions_for_upcoming_games()

        assert result == [good_pred]

    @pytest.mark.anyio
    async def test_skips_structurally_invalid_prediction(self, repo):
        r, session = repo
        good_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC) - timedelta(minutes=5),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": "2026-03-30T05:20:00Z"},
        )
        bad_newer_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=-6.0,
            predicted_away_fg=1.0,
            predicted_home_1h=-3.0,
            predicted_away_1h=1.0,
            fg_spread=-7.0,
            fg_total=-5.0,
            h1_spread=-4.0,
            h1_total=-2.0,
            odds_sourced={"captured_at": "2026-03-30T05:25:00Z"},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [bad_newer_pred, good_pred]
        session.execute = AsyncMock(return_value=mock_result)

        result = await r.get_latest_predictions_for_upcoming_games()

        assert result == [good_pred]

    @pytest.mark.anyio
    async def test_empty(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_predictions_for_upcoming_games()
        assert result == []

    @pytest.mark.anyio
    async def test_drops_game_when_all_rows_are_invalid(self, repo):
        r, session = repo
        invalid_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=-6.0,
            predicted_away_fg=1.0,
            predicted_home_1h=-3.0,
            predicted_away_1h=1.0,
            fg_spread=-7.0,
            fg_total=-5.0,
            h1_spread=-4.0,
            h1_total=-2.0,
            odds_sourced={"captured_at": "2026-03-30T05:25:00Z"},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [invalid_pred]
        session.execute = AsyncMock(return_value=mock_result)

        result = await r.get_latest_predictions_for_upcoming_games()

        assert result == []

    @pytest.mark.anyio
    async def test_drops_implausible_low_score_prediction(self, repo):
        r, session = repo
        implausible_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=4.3,
            predicted_away_fg=7.8,
            predicted_home_1h=2.1,
            predicted_away_1h=5.3,
            fg_spread=-3.5,
            fg_total=12.1,
            h1_spread=-3.2,
            h1_total=7.3,
            odds_sourced={"captured_at": "2026-03-30T05:25:00Z"},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [implausible_pred]
        session.execute = AsyncMock(return_value=mock_result)

        result = await r.get_latest_predictions_for_upcoming_games()

        assert result == []

    @pytest.mark.anyio
    async def test_filters_to_linked_upcoming_games(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        await r.get_latest_predictions_for_upcoming_games()

        stmt = session.execute.await_args.args[0]
        sql = str(stmt.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
        assert "games.status = 'NS'" in sql
        assert "games.odds_api_id IS NOT NULL" in sql


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
        pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": "2026-03-30T05:20:00Z"},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [pred]
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_prediction_for_game(1)
        assert result is pred

    @pytest.mark.anyio
    async def test_not_found(self, repo):
        r, session = repo
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)
        result = await r.get_latest_prediction_for_game(999)
        assert result is None

    @pytest.mark.anyio
    async def test_prefers_odds_backed_prediction_for_single_game(self, repo):
        r, session = repo
        good_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC) - timedelta(minutes=5),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": "2026-03-30T05:20:00Z"},
        )
        bad_newer_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=110.0,
            predicted_away_fg=105.0,
            predicted_home_1h=55.0,
            predicted_away_1h=52.0,
            fg_spread=5.0,
            fg_total=215.0,
            h1_spread=3.0,
            h1_total=107.0,
            odds_sourced={"captured_at": None},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [bad_newer_pred, good_pred]
        session.execute = AsyncMock(return_value=mock_result)

        result = await r.get_latest_prediction_for_game(1)

        assert result is good_pred

    @pytest.mark.anyio
    async def test_returns_none_when_all_single_game_rows_are_invalid(self, repo):
        r, session = repo
        invalid_pred = MagicMock(
            game_id=1,
            predicted_at=datetime.now(UTC),
            predicted_home_fg=-6.0,
            predicted_away_fg=1.0,
            predicted_home_1h=-3.0,
            predicted_away_1h=1.0,
            fg_spread=-7.0,
            fg_total=-5.0,
            h1_spread=-4.0,
            h1_total=-2.0,
            odds_sourced={"captured_at": "2026-03-30T05:25:00Z"},
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [invalid_pred]
        session.execute = AsyncMock(return_value=mock_result)

        result = await r.get_latest_prediction_for_game(1)

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
        from datetime import UTC, datetime

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
