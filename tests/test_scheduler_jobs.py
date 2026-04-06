"""Tests for scheduler job bodies.

Each job function is async and relies on external clients + DB sessions.
We mock all I/O and verify the orchestration logic.

NOTE: Job code now lives in submodules under src.data.jobs.*,
so we patch at the *source* module paths.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.scheduler import (
    check_prediction_drift,
    daily_retrain,
    db_maintenance,
    fill_clv,
    poll_1h_odds,
    poll_fg_odds,
    poll_injuries,
    poll_player_props,
    poll_scores_and_box,
    poll_stats,
    prune_old_odds,
    sync_events_to_games,
)

# Session factory lives in each submodule's namespace
_SF_POLL = "src.data.jobs.polling.async_session_factory"
_SF_PRED = "src.data.jobs.predictions.async_session_factory"
_SF_MAINT = "src.data.jobs.maintenance.async_session_factory"


# -- poll_fg_odds --


class TestPollFgOdds:
    @patch("src.data.jobs.polling.sync_events_to_games", new_callable=AsyncMock)
    @patch("src.data.circuit_breaker.CircuitBreaker.record_success")
    @patch("src.data.circuit_breaker.CircuitBreaker.should_skip", return_value=False)
    @patch("src.data.odds_client.OddsClient")
    @patch(_SF_POLL)
    async def test_persists_odds(self, mock_sf, mock_cls, mock_skip, mock_success, mock_sync):
        mock_client = mock_cls.return_value
        mock_client.fetch_odds = AsyncMock(return_value=[{"bookmakers": []}])
        mock_client.persist_odds = AsyncMock(return_value=5)

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_fg_odds()

        mock_sync.assert_awaited_once()
        mock_client.fetch_odds.assert_awaited_once()
        mock_client.persist_odds.assert_awaited_once()
        mock_success.assert_called()

    @patch("src.data.circuit_breaker.CircuitBreaker.should_skip", return_value=True)
    async def test_skips_when_circuit_open(self, mock_skip):
        await poll_fg_odds()


# -- poll_1h_odds --


class TestPoll1hOdds:
    @patch("src.data.jobs.polling.sync_events_to_games", new_callable=AsyncMock)
    @patch("src.data.circuit_breaker.CircuitBreaker.record_success")
    @patch("src.data.circuit_breaker.CircuitBreaker.should_skip", return_value=False)
    @patch("src.data.odds_client.OddsClient")
    @patch(_SF_POLL)
    async def test_fetches_event_odds(
        self,
        mock_sf,
        mock_cls,
        mock_skip,
        mock_success,
        mock_sync,
    ):
        mock_client = mock_cls.return_value
        mock_client.fetch_events = AsyncMock(
            return_value=[
                {"id": "ev1"},
                {"id": "ev2"},
            ]
        )
        mock_client.fetch_event_odds = AsyncMock(return_value={"bookmakers": [{}]})
        mock_client.persist_odds = AsyncMock(return_value=1)
        mock_client._should_skip = MagicMock(return_value=False)

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_1h_odds()

        mock_sync.assert_awaited_once()
        mock_client.fetch_events.assert_awaited_once()
        assert mock_client.fetch_event_odds.await_count >= 1

    @patch("src.data.circuit_breaker.CircuitBreaker.should_skip", return_value=False)
    @patch("src.data.odds_client.OddsClient")
    @patch(_SF_POLL)
    async def test_handles_empty_events(self, mock_sf, mock_cls, mock_skip):
        mock_client = mock_cls.return_value
        mock_client.fetch_events = AsyncMock(return_value=[])
        mock_client._should_skip = MagicMock(return_value=False)

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_1h_odds()

    @patch("src.data.circuit_breaker.CircuitBreaker.should_skip", return_value=True)
    async def test_skips_when_circuit_open(self, mock_skip):
        await poll_1h_odds()


# -- poll_player_props --


class TestPollPlayerProps:
    @patch("src.data.odds_client.OddsClient")
    @patch(_SF_POLL)
    async def test_fetches_player_props(self, mock_sf, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.fetch_events = AsyncMock(return_value=[{"id": "ev1"}])
        mock_client.fetch_player_props = AsyncMock(return_value={"bookmakers": [{}]})
        mock_client.persist_odds = AsyncMock(return_value=1)
        mock_client._should_skip = MagicMock(return_value=False)

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_player_props()

        mock_client.fetch_player_props.assert_awaited_once()


# -- poll_stats --


class TestPollStats:
    @patch("src.data.circuit_breaker.CircuitBreaker.record_success")
    @patch("src.data.circuit_breaker.CircuitBreaker.should_skip", return_value=False)
    @patch("src.data.basketball_client.BasketballClient")
    @patch(_SF_POLL)
    async def test_fetches_and_persists_stats(self, mock_sf, mock_cls, mock_skip, mock_success):
        mock_client = mock_cls.return_value
        mock_client.fetch_games = AsyncMock(return_value=[{"id": 1}])
        mock_client.persist_games = AsyncMock(return_value=1)
        mock_client.fetch_team_stats = AsyncMock(return_value={"games": {}})
        mock_client.persist_team_season_stats = AsyncMock()

        mock_db = AsyncMock()
        team_result = MagicMock()
        team_result.fetchall.return_value = [(1,), (2,)]
        count_result = MagicMock()
        count_result.scalar.return_value = 30

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return team_result
            return count_result

        mock_db.execute = mock_exec
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_stats()

        mock_success.assert_called()


# -- poll_scores_and_box --


class TestPollScoresAndBox:
    @patch("src.data.basketball_client.BasketballClient")
    @patch(_SF_POLL)
    async def test_fetches_missing_box_scores(self, mock_sf, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.fetch_player_stats = AsyncMock(return_value=[{"player": {"id": 1}}])
        mock_client.persist_player_game_stats = AsyncMock(return_value=1)

        mock_db = AsyncMock()
        games_result = MagicMock()
        games_result.fetchall.return_value = [(100,), (101,)]
        mock_db.execute = AsyncMock(return_value=games_result)
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_scores_and_box()

        assert mock_client.fetch_player_stats.await_count >= 1

    @patch("src.data.basketball_client.BasketballClient")
    @patch(_SF_POLL)
    async def test_no_missing_games(self, mock_sf, mock_cls):
        mock_db = AsyncMock()
        games_result = MagicMock()
        games_result.fetchall.return_value = []
        mock_db.execute = AsyncMock(return_value=games_result)
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_scores_and_box()


# -- poll_injuries --


class TestPollInjuries:
    @patch("src.data.basketball_client.BasketballClient")
    @patch(_SF_POLL)
    async def test_persists_injuries(self, mock_sf, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.fetch_injuries = AsyncMock(return_value=[{"player": {"id": 1}}])
        mock_client.persist_injuries = AsyncMock(return_value=5)

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_injuries()

        mock_client.persist_injuries.assert_awaited_once()

    @patch("src.data.basketball_client.BasketballClient")
    @patch(_SF_POLL)
    async def test_handles_no_injuries(self, mock_sf, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.fetch_injuries = AsyncMock(return_value=[])

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await poll_injuries()


# -- sync_events_to_games --


class TestSyncEventsToGames:
    @patch("src.data.odds_client.OddsClient")
    @patch(_SF_POLL)
    async def test_maps_events_to_games(self, mock_sf, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.fetch_events = AsyncMock(
            return_value=[
                {
                    "id": "odds1",
                    "commence_time": "2025-01-15T19:00:00Z",
                    "home_team": "Boston Celtics",
                },
            ]
        )

        mock_game = SimpleNamespace(odds_api_id=None)
        mock_db = AsyncMock()
        game_result = MagicMock()
        game_result.scalar_one_or_none.return_value = mock_game
        mock_db.execute = AsyncMock(return_value=game_result)
        mock_db.commit = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await sync_events_to_games()

        mock_db.commit.assert_awaited()


# -- fill_clv --


class TestFillClv:
    @patch(_SF_MAINT)
    async def test_fills_clv_for_finished_preds(self, mock_sf):
        pred = SimpleNamespace(
            game_id=1,
            opening_spread=-5.5,
            opening_total=220.0,
            closing_spread=None,
            closing_total=None,
            clv_spread=None,
            clv_total=None,
            game=SimpleNamespace(
                id=1,
                home_team=SimpleNamespace(name="BOS"),
            ),
        )

        odds_snap = SimpleNamespace(
            market="spreads",
            point=-6.0,
            outcome_name="BOS",
            captured_at=datetime(2025, 1, 15, tzinfo=UTC),
        )

        mock_db = AsyncMock()
        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            if call_n["n"] == 1:
                result.scalars.return_value.all.return_value = [pred]
            elif call_n["n"] == 2:
                result.scalars.return_value.all.return_value = [odds_snap]
            else:
                result.scalars.return_value.all.return_value = []
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await fill_clv()

    @patch(_SF_MAINT)
    async def test_no_preds_noop(self, mock_sf):
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await fill_clv()


# -- check_prediction_drift --


class TestCheckPredictionDrift:
    @patch(_SF_PRED)
    async def test_no_drift(self, mock_sf):
        mock_db = AsyncMock()
        result_30d = MagicMock()
        result_30d.all.return_value = [(105.0, 100.0)] * 25
        result_7d = MagicMock()
        result_7d.all.return_value = [(106.0, 101.0)] * 10

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return result_30d
            return result_7d

        mock_db.execute = mock_exec
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await check_prediction_drift()

    @patch(_SF_PRED)
    async def test_insufficient_data(self, mock_sf):
        mock_db = AsyncMock()
        result = MagicMock()
        result.all.return_value = [(100.0, 100.0)] * 3
        mock_db.execute = AsyncMock(return_value=result)
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await check_prediction_drift()


# -- prune_old_odds --


class TestPruneOldOdds:
    @patch(_SF_MAINT)
    async def test_deletes_old_odds(self, mock_sf):
        mock_db = AsyncMock()
        delete_result = MagicMock()
        delete_result.rowcount = 150
        mock_db.execute = AsyncMock(return_value=delete_result)
        mock_db.commit = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await prune_old_odds()
        mock_db.commit.assert_awaited()


# -- db_maintenance --


class TestDbMaintenance:
    @patch(_SF_MAINT)
    async def test_runs_analyze(self, mock_sf):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await db_maintenance()
        assert mock_db.execute.await_count == 5
        mock_db.commit.assert_awaited_once()


# -- daily_retrain --


class TestDailyRetrain:
    @patch("src.models.trainer.ModelTrainer")
    @patch(_SF_POLL)
    async def test_invokes_trainer(self, mock_sf, mock_cls):
        mock_trainer = mock_cls.return_value
        mock_trainer.train = AsyncMock(return_value={"mae": 8.0})

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await daily_retrain()

        mock_trainer.train.assert_awaited_once_with(mock_db)

    @patch("src.notifications.teams.send_alert", new_callable=AsyncMock)
    @patch("src.models.trainer.ModelTrainer")
    @patch(_SF_POLL)
    async def test_sends_alert_on_failure(self, mock_sf, mock_cls, mock_alert):
        mock_trainer = mock_cls.return_value
        mock_trainer.train = AsyncMock(side_effect=RuntimeError("boom"))

        mock_db = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        await daily_retrain()
        mock_alert.assert_awaited_once()
