"""Tests for scheduler poll functions and sync/CLV/publish flows."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

_SCHED = "src.data.scheduler"


def _mock_session():
    """Create a mock async session factory context manager."""
    mock_sf = MagicMock()
    mock_db = AsyncMock()
    mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_db)
    mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_sf, mock_db


# ── poll_fg_odds ─────────────────────────────────────────────────


class TestPollFgOdds:
    @pytest.mark.anyio
    async def test_success_with_data(self):
        from src.data.scheduler import poll_fg_odds

        mock_sf, mock_db = _mock_session()
        mock_client = MagicMock()
        mock_client.fetch_odds = AsyncMock(return_value=[{"id": "e1"}])
        mock_client.persist_odds = AsyncMock()

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
        ):
            mock_breaker.should_skip.return_value = False
            await poll_fg_odds()
        mock_client.persist_odds.assert_awaited_once()
        mock_breaker.record_success.assert_called_once()


# ── poll_1h_odds ─────────────────────────────────────────────────


class TestPoll1hOdds:
    @pytest.mark.anyio
    async def test_success_with_events(self):
        from src.data.scheduler import poll_1h_odds

        mock_sf, mock_db = _mock_session()
        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(
            return_value=[{"id": "ev1"}, {"id": "ev2"}, {}]  # 3rd has no id
        )
        mock_client.fetch_event_odds = AsyncMock(
            return_value={"bookmakers": [{"key": "dk"}]}
        )
        mock_client.persist_odds = AsyncMock()
        mock_client._should_skip = MagicMock(return_value=False)

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock) as mock_sync,
        ):
            mock_breaker.should_skip.return_value = False
            await poll_1h_odds()
        mock_sync.assert_awaited_once()
        assert mock_client.persist_odds.call_count == 2
        mock_breaker.record_success.assert_called_once()

    @pytest.mark.anyio
    async def test_quota_break(self):
        """Stops early when quota is low."""
        from src.data.scheduler import poll_1h_odds

        mock_sf, mock_db = _mock_session()
        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(return_value=[{"id": "ev1"}, {"id": "ev2"}])
        mock_client._should_skip = MagicMock(return_value=True)
        mock_client.fetch_event_odds = AsyncMock()
        mock_client.persist_odds = AsyncMock()

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
        ):
            mock_breaker.should_skip.return_value = False
            await poll_1h_odds()
        mock_client.fetch_event_odds.assert_not_awaited()

    @pytest.mark.anyio
    async def test_exception_records_failure(self):
        from src.data.scheduler import poll_1h_odds

        with (
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient") as mock_cls,
            patch(f"{_SCHED}._record_failure", new_callable=AsyncMock),
        ):
            mock_breaker.should_skip.return_value = False
            mock_cls.return_value.fetch_events = AsyncMock(side_effect=RuntimeError("boom"))
            await poll_1h_odds()
        mock_breaker.record_failure.assert_called_once()


# ── poll_player_props ────────────────────────────────────────────


class TestPollPlayerProps:
    @pytest.mark.anyio
    async def test_breaker_skip(self):
        from src.data.scheduler import poll_player_props

        with patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker:
            mock_breaker.should_skip.return_value = True
            await poll_player_props()

    @pytest.mark.anyio
    async def test_success_with_props(self):
        from src.data.scheduler import poll_player_props

        mock_sf, mock_db = _mock_session()
        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(return_value=[{"id": "e1"}, {}])
        mock_client._should_skip = MagicMock(return_value=False)
        mock_client.fetch_player_props = AsyncMock(
            return_value={"bookmakers": [{"key": "fd"}]}
        )
        mock_client.persist_odds = AsyncMock()

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
        ):
            mock_breaker.should_skip.return_value = False
            await poll_player_props()
        mock_client.persist_odds.assert_awaited_once()
        mock_breaker.record_success.assert_called_once()

    @pytest.mark.anyio
    async def test_quota_break(self):
        from src.data.scheduler import poll_player_props

        mock_sf, mock_db = _mock_session()
        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(return_value=[{"id": "e1"}])
        mock_client._should_skip = MagicMock(return_value=True)
        mock_client.fetch_player_props = AsyncMock()
        mock_client.persist_odds = AsyncMock()

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
        ):
            mock_breaker.should_skip.return_value = False
            await poll_player_props()
        mock_client.fetch_player_props.assert_not_awaited()

    @pytest.mark.anyio
    async def test_exception_records_failure(self):
        from src.data.scheduler import poll_player_props

        with (
            patch("src.data.circuit_breaker.odds_api_breaker") as mock_breaker,
            patch("src.data.odds_client.OddsClient") as mock_cls,
            patch(f"{_SCHED}._record_failure", new_callable=AsyncMock),
        ):
            mock_breaker.should_skip.return_value = False
            mock_cls.return_value.fetch_events = AsyncMock(side_effect=RuntimeError("x"))
            await poll_player_props()
        mock_breaker.record_failure.assert_called_once()


# ── poll_stats ───────────────────────────────────────────────────


class TestPollStats:
    @pytest.mark.anyio
    async def test_breaker_skip(self):
        from src.data.scheduler import poll_stats

        with patch("src.data.circuit_breaker.basketball_api_breaker") as mock_b:
            mock_b.should_skip.return_value = True
            await poll_stats()

    @pytest.mark.anyio
    async def test_success_with_data(self):
        from src.data.scheduler import poll_stats

        mock_sf, mock_db = _mock_session()
        mock_client = MagicMock()
        mock_client.fetch_games = AsyncMock(return_value=[{"id": 1}])
        mock_client.persist_games = AsyncMock(return_value=3)
        mock_client.fetch_team_stats = AsyncMock(return_value={"ppg": 110})
        mock_client.persist_team_season_stats = AsyncMock()

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            sql = str(stmt).lower()
            result = MagicMock()
            if "teams" in sql and "count" not in sql:
                result.fetchall.return_value = [(1,), (2,)]
                return result
            # count query for team_count
            result.scalar.return_value = 25  # < 30 to trigger warning
            return result

        mock_db.execute = mock_exec

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.circuit_breaker.basketball_api_breaker") as mock_b,
            patch("src.data.basketball_client.BasketballClient", return_value=mock_client),
            patch(f"{_SCHED}.reconcile_duplicate_games", new_callable=AsyncMock, return_value=2) as mock_reconcile,
        ):
            mock_b.should_skip.return_value = False
            await poll_stats()
        # 3 dates × 1 game each = 3 calls
        assert mock_client.fetch_games.call_count == 3
        mock_reconcile.assert_awaited_once_with(mock_db)
        mock_b.record_success.assert_called_once()


# ── poll_scores_and_box ──────────────────────────────────────────


class TestPollScoresAndBox:
    @pytest.mark.anyio
    async def test_no_missing_games(self):
        from src.data.scheduler import poll_scores_and_box

        mock_sf, mock_db = _mock_session()
        result = MagicMock()
        result.fetchall.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.basketball_client.BasketballClient") as _cls,
        ):
            await poll_scores_and_box()

    @pytest.mark.anyio
    async def test_fetches_box_scores(self):
        from src.data.scheduler import poll_scores_and_box

        mock_sf, mock_db = _mock_session()
        result = MagicMock()
        result.fetchall.return_value = [(100,), (200,)]
        mock_db.execute = AsyncMock(return_value=result)

        mock_client = MagicMock()
        mock_client.fetch_player_stats = AsyncMock(return_value=[{"player": "A"}])
        mock_client.persist_player_game_stats = AsyncMock()

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.basketball_client.BasketballClient", return_value=mock_client),
        ):
            await poll_scores_and_box()
        assert mock_client.persist_player_game_stats.call_count == 2

    @pytest.mark.anyio
    async def test_box_score_failure_continues(self):
        from src.data.scheduler import poll_scores_and_box

        mock_sf, mock_db = _mock_session()
        result = MagicMock()
        result.fetchall.return_value = [(100,)]
        mock_db.execute = AsyncMock(return_value=result)

        mock_client = MagicMock()
        mock_client.fetch_player_stats = AsyncMock(side_effect=RuntimeError("API err"))

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.basketball_client.BasketballClient", return_value=mock_client),
        ):
            await poll_scores_and_box()  # should not raise

    @pytest.mark.anyio
    async def test_outer_exception_handled(self):
        from src.data.scheduler import poll_scores_and_box

        with patch(f"{_SCHED}.async_session_factory") as mock_sf:
            mock_sf.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("db"))
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await poll_scores_and_box()  # caught


# ── sync_events_to_games ─────────────────────────────────────────


class TestSyncEventsToGames:
    @pytest.mark.anyio
    async def test_fallback_match_by_home_team(self):
        from src.data.scheduler import sync_events_to_games

        mock_sf, mock_db = _mock_session()
        mock_game = SimpleNamespace(odds_api_id=None)

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            sql = str(stmt).lower()
            result = MagicMock()
            if "where games.odds_api_id" in sql:
                result.scalar_one_or_none.return_value = None
                return result
            if "between" in sql:
                # fallback: same home team within ±12 hours
                result.scalar_one_or_none.return_value = mock_game
                return result
            if "where games.commence_time" in sql:
                # exact commence_time match
                result.scalar_one_or_none.return_value = None
                return result
            if "select teams" in sql:
                result.all.return_value = [(1, "TeamA"), (2, "TeamB")]
                return result
            result.scalar_one_or_none.return_value = None
            result.all.return_value = []
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()

        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(
            return_value=[{
                "id": "evt1",
                "commence_time": "2024-10-01T00:00:00Z",
                "home_team": "TeamA",
                "away_team": "TeamB",
            }]
        )

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.parse_api_datetime", return_value=datetime(2024, 10, 1)),
        ):
            await sync_events_to_games()
        assert mock_game.odds_api_id == "evt1"

    @pytest.mark.anyio
    async def test_unmatched_events_are_skipped_without_synthetic_creation(self):
        from src.data.scheduler import sync_events_to_games

        mock_sf, mock_db = _mock_session()

        async def mock_exec(stmt, *a, **kw):
            sql = str(stmt).lower()
            result = MagicMock()
            if "select teams" in sql:
                result.all.return_value = [(1, "Lakers"), (2, "Celtics")]
                return result
            # Return None for all game lookups -> unmatched event should be skipped.
            result.scalar_one_or_none.return_value = None
            result.all.return_value = []
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()

        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(
            return_value=[{
                "id": "odds_evt_99",
                "commence_time": "2024-11-01T00:00:00Z",
                "home_team": "Lakers",
                "away_team": "Celtics",
            }]
        )

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.parse_api_datetime", return_value=datetime(2024, 11, 1)),
        ):
            await sync_events_to_games()
        mock_db.add.assert_not_called()

    @pytest.mark.anyio
    async def test_team_lookup_failed(self):
        from src.data.scheduler import sync_events_to_games

        mock_sf, mock_db = _mock_session()

        async def mock_exec(stmt, *a, **kw):
            sql = str(stmt).lower()
            result = MagicMock()
            if "select teams" in sql:
                result.all.return_value = [(1, "Lakers")]  # Missing Celtics
                return result
            result.scalar_one_or_none.return_value = None
            result.all.return_value = []
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()

        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(
            return_value=[{
                "id": "evt1",
                "commence_time": "2024-11-01T00:00:00Z",
                "home_team": "Lakers",
                "away_team": "Celtics",  # Not in team_by_name
            }]
        )

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.parse_api_datetime", return_value=datetime(2024, 11, 1)),
        ):
            await sync_events_to_games()
        mock_db.add.assert_not_called()

    @pytest.mark.anyio
    async def test_exception_handled(self):
        from src.data.scheduler import sync_events_to_games

        with patch("src.data.odds_client.OddsClient") as mock_cls:
            mock_cls.return_value.fetch_events = AsyncMock(side_effect=RuntimeError("fail"))
            await sync_events_to_games()

    @pytest.mark.anyio
    async def test_existing_synthetic_reconciles_to_real_game(self):
        from src.data.scheduler import sync_events_to_games

        mock_sf, mock_db = _mock_session()
        synthetic_game = SimpleNamespace(id=-100, odds_api_id="evt1")
        real_game = SimpleNamespace(id=500, odds_api_id=None)

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            sql = str(stmt).lower()
            if "select teams" in sql:
                result.all.return_value = [(1, "Lakers"), (2, "Celtics")]
                return result
            if "where games.odds_api_id" in sql:
                result.scalar_one_or_none.return_value = synthetic_game
                return result
            result.scalar_one_or_none.return_value = None
            result.all.return_value = []
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()

        mock_client = MagicMock()
        mock_client.fetch_events = AsyncMock(
            return_value=[{
                "id": "evt1",
                "commence_time": "2024-11-01T00:00:00Z",
                "home_team": "Lakers",
                "away_team": "Celtics",
            }]
        )

        with (
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.data.odds_client.OddsClient", return_value=mock_client),
            patch(f"{_SCHED}.parse_api_datetime", return_value=datetime(2024, 11, 1)),
            patch(f"{_SCHED}._find_matching_game", new_callable=AsyncMock, return_value=real_game),
            patch(f"{_SCHED}._merge_game_records", new_callable=AsyncMock, return_value=True) as mock_merge,
        ):
            await sync_events_to_games()
        mock_merge.assert_awaited_once_with(mock_db, synthetic_game, real_game)


# ── fill_clv ─────────────────────────────────────────────────────


class TestFillClv:
    @pytest.mark.anyio
    async def test_fills_spread_and_total(self):
        from src.data.scheduler import fill_clv

        mock_sf, mock_db = _mock_session()

        # Build mock prediction with game
        home_team = SimpleNamespace(name="Lakers")
        game = SimpleNamespace(home_team=home_team)
        pred = MagicMock()
        pred.game = game
        pred.game_id = 1
        pred.opening_spread = -3.0
        pred.opening_total = 220.0
        pred.closing_spread = None

        # Odds snapshots
        snap_spread = SimpleNamespace(
            market="spreads", point=-4.5, outcome_name="Lakers", captured_at=datetime.now()
        )
        snap_total = SimpleNamespace(
            market="totals", point=222.0, outcome_name="Over", captured_at=datetime.now()
        )

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            sql = str(stmt).lower()
            if "prediction" in sql and "closing_spread" in sql:
                result.scalars.return_value.all.return_value = [pred]
                return result
            if "odds_snapshot" in sql or "oddssnapshot" in sql:
                result.scalars.return_value.all.return_value = [snap_spread, snap_total]
                return result
            result.scalars.return_value.all.return_value = []
            return result

        mock_db.execute = mock_exec
        mock_db.commit = AsyncMock()

        with patch(f"{_SCHED}.async_session_factory", mock_sf):
            await fill_clv()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_no_predictions_returns_early(self):
        from src.data.scheduler import fill_clv

        mock_sf, mock_db = _mock_session()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        with patch(f"{_SCHED}.async_session_factory", mock_sf):
            await fill_clv()

    @pytest.mark.anyio
    async def test_exception_handled(self):
        from src.data.scheduler import fill_clv

        with patch(f"{_SCHED}.async_session_factory") as mock_sf:
            mock_sf.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("db"))
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)
            await fill_clv()


class TestGameReconciliationHelpers:
    @pytest.mark.anyio
    async def test_reconcile_duplicate_games_merges_recent_synthetic_rows(self):
        from src.data.scheduler import reconcile_duplicate_games

        mock_db = AsyncMock()
        synthetic_game = SimpleNamespace(
            id=-50,
            home_team_id=1,
            away_team_id=2,
            commence_time=datetime(2026, 3, 29, 19, 0),
        )
        result = MagicMock()
        result.scalars.return_value.all.return_value = [synthetic_game]
        mock_db.execute = AsyncMock(return_value=result)

        with (
            patch(f"{_SCHED}._find_matching_game", new_callable=AsyncMock, return_value=SimpleNamespace(id=10)),
            patch(f"{_SCHED}._merge_game_records", new_callable=AsyncMock, return_value=True) as mock_merge,
        ):
            reconciled = await reconcile_duplicate_games(mock_db, lookback_days=None)

        assert reconciled == 1
        mock_merge.assert_awaited_once()

    @pytest.mark.anyio
    async def test_merge_game_records_prefers_odds_backed_prediction(self):
        from src.data.scheduler import _merge_game_records

        mock_db = AsyncMock()
        source_game = SimpleNamespace(id=-10, odds_api_id="evt-1")
        target_game = SimpleNamespace(id=100, odds_api_id=None)

        def _prediction(**overrides):
            base = {
                "game_id": -10,
                "model_version": "v1",
                "predicted_home_fg": 110.0,
                "predicted_away_fg": 104.0,
                "predicted_home_1h": 54.0,
                "predicted_away_1h": 50.0,
                "fg_spread": 6.0,
                "fg_total": 214.0,
                "fg_home_ml_prob": 0.63,
                "h1_spread": 4.0,
                "h1_total": 104.0,
                "h1_home_ml_prob": 0.61,
                "opening_spread": -5.5,
                "opening_total": 219.5,
                "closing_spread": None,
                "closing_total": None,
                "clv_spread": None,
                "clv_total": None,
                "odds_sourced": None,
                "predicted_at": datetime(2026, 3, 29, 8, 0),
            }
            base.update(overrides)
            return SimpleNamespace(**base)

        target_pred = _prediction(
            game_id=100,
            opening_spread=None,
            opening_total=None,
            closing_spread=-4.0,
            clv_spread=1.0,
            predicted_at=datetime(2026, 3, 29, 7, 0),
        )
        source_pred = _prediction(
            odds_sourced={"captured_at": "2026-03-29T12:00:00Z"},
            predicted_at=datetime(2026, 3, 29, 9, 0),
        )

        target_pred_result = MagicMock()
        target_pred_result.scalars.return_value.all.return_value = [target_pred]
        source_pred_result = MagicMock()
        source_pred_result.scalars.return_value.all.return_value = [source_pred]
        target_players_result = MagicMock()
        target_players_result.scalars.return_value.all.return_value = [7]
        noop_result = MagicMock()

        mock_db.execute = AsyncMock(
            side_effect=[
                target_pred_result,
                source_pred_result,
                target_players_result,
                noop_result,
                noop_result,
                noop_result,
                noop_result,
                noop_result,
            ]
        )
        mock_db.delete = AsyncMock()
        mock_db.flush = AsyncMock()

        merged = await _merge_game_records(mock_db, source_game, target_game)

        assert merged is True
        assert target_game.odds_api_id == "evt-1"
        assert source_game.odds_api_id is None
        assert target_pred.odds_sourced == {"captured_at": "2026-03-29T12:00:00Z"}
        assert target_pred.opening_spread == -5.5
        assert target_pred.closing_spread == -4.0
        assert mock_db.delete.await_count == 2


# ── pregame_check trigger ────────────────────────────────────────


class TestPregameCheckTrigger:
    @pytest.mark.anyio
    async def test_pregame_triggers_publish(self):
        import src.data.scheduler as sched
        from src.data.scheduler import pregame_check

        original = sched._pregame_published_date
        sched._pregame_published_date = None
        try:
            et = ZoneInfo("US/Eastern")
            now = datetime(2026, 3, 29, 19, 0, tzinfo=et)
            # A game starting 30 minutes from now (within default 60-min lead)
            game_ct_utc = (now + timedelta(minutes=30)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

            mock_sf, mock_db = _mock_session()
            result = MagicMock()
            result.scalar_one_or_none.return_value = game_ct_utc
            mock_db.execute = AsyncMock(return_value=result)

            class _FixedDateTime(datetime):
                @classmethod
                def now(cls, tz=None):
                    if tz is None:
                        return now.replace(tzinfo=None)
                    return now.astimezone(tz)

            with (
                patch(f"{_SCHED}.async_session_factory", mock_sf),
                patch(f"{_SCHED}.generate_predictions_and_publish", new_callable=AsyncMock) as mock_pub,
                patch(f"{_SCHED}.get_settings") as mock_s,
                patch(f"{_SCHED}.datetime", _FixedDateTime),
            ):
                mock_s.return_value.pregame_lead_minutes = 60
                await pregame_check()
            mock_pub.assert_awaited_once()
        finally:
            sched._pregame_published_date = original

    @pytest.mark.anyio
    async def test_game_tomorrow_skipped(self):
        import src.data.scheduler as sched
        from src.data.scheduler import pregame_check

        original = sched._pregame_published_date
        sched._pregame_published_date = None
        try:
            et = ZoneInfo("US/Eastern")
            tomorrow_utc = (datetime.now(et) + timedelta(days=1)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

            mock_sf, mock_db = _mock_session()
            result = MagicMock()
            result.scalar_one_or_none.return_value = tomorrow_utc
            mock_db.execute = AsyncMock(return_value=result)

            with (
                patch(f"{_SCHED}.async_session_factory", mock_sf),
                patch(f"{_SCHED}.generate_predictions_and_publish", new_callable=AsyncMock) as mock_pub,
            ):
                await pregame_check()
            mock_pub.assert_not_awaited()
        finally:
            sched._pregame_published_date = original


# ── generate_predictions_and_publish ─────────────────────────────


class TestPublishFlow:
    @pytest.mark.anyio
    async def test_no_predictions_no_games(self):
        """0 NS games + 0 predictions → logs 'nothing to predict'."""
        from src.data.scheduler import generate_predictions_and_publish

        mock_sf, mock_db = _mock_session()
        # ns_count = 0
        ns_result = MagicMock()
        ns_result.scalar.return_value = 0
        mock_db.execute = AsyncMock(return_value=ns_result)

        mock_predictor = MagicMock(is_ready=True)
        mock_predictor.predict_upcoming = AsyncMock(return_value=[])

        with (
            patch(f"{_SCHED}.poll_stats", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_SCHED}.get_settings") as mock_s,
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.models.features.reset_elo_cache"),
        ):
            mock_s.return_value = MagicMock()
            await generate_predictions_and_publish()

    @pytest.mark.anyio
    async def test_data_loss_alert_sent(self):
        """NS games > 0 but 0 predictions → sends DATA LOSS alert."""
        from src.data.scheduler import generate_predictions_and_publish

        mock_sf, mock_db = _mock_session()
        ns_result = MagicMock()
        ns_result.scalar.return_value = 5
        mock_db.execute = AsyncMock(return_value=ns_result)

        mock_predictor = MagicMock(is_ready=True)
        mock_predictor.predict_upcoming = AsyncMock(return_value=[])

        with (
            patch(f"{_SCHED}.poll_stats", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_SCHED}.get_settings") as mock_s,
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.models.features.reset_elo_cache"),
            patch("src.notifications.teams.send_alert", new_callable=AsyncMock) as mock_alert,
        ):
            mock_s.return_value = MagicMock()
            await generate_predictions_and_publish()
        mock_alert.assert_awaited_once()

    @pytest.mark.anyio
    async def test_webhook_publish(self):
        """Predictions published via webhook when configured."""
        from src.data.scheduler import generate_predictions_and_publish

        mock_sf, mock_db = _mock_session()

        mock_pred = MagicMock()
        mock_pred.game_id = 1
        mock_game = MagicMock()
        mock_game.id = 1

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            sql = str(stmt).lower()
            if "count" in sql:
                result.scalar.return_value = 1
                return result
            if "max" in sql:
                result.scalar_one_or_none.return_value = datetime.now()
                return result
            result.scalars.return_value.all.return_value = [mock_game]
            return result

        mock_db.execute = mock_exec

        mock_predictor = MagicMock(is_ready=True)
        mock_predictor.predict_upcoming = AsyncMock(return_value=[mock_pred])

        mock_settings = MagicMock()
        mock_settings.teams_team_id = ""
        mock_settings.teams_channel_id = ""
        mock_settings.teams_webhook_url = "https://webhook.example.com"
        mock_settings.api_base_url = "https://api.example.com"
        mock_settings.teams_max_games_per_message = 10

        with (
            patch(f"{_SCHED}.poll_stats", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_SCHED}.get_settings", return_value=mock_settings),
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.models.features.reset_elo_cache"),
            patch("src.notifications.teams.build_teams_card", return_value={"body": []}),
            patch("src.notifications.teams.send_card_to_teams", new_callable=AsyncMock) as mock_send,
        ):
            await generate_predictions_and_publish()
        mock_send.assert_awaited_once()

    @pytest.mark.anyio
    async def test_graph_api_publish(self):
        """Predictions published via Graph API when team_id + channel_id set."""
        from src.data.scheduler import generate_predictions_and_publish

        mock_sf, mock_db = _mock_session()

        mock_pred = MagicMock()
        mock_pred.game_id = 1
        mock_game = MagicMock()
        mock_game.id = 1

        call_n = {"n": 0}

        async def mock_exec(stmt, *a, **kw):
            call_n["n"] += 1
            result = MagicMock()
            sql = str(stmt).lower()
            if "count" in sql:
                result.scalar.return_value = 1
                return result
            if "max" in sql:
                result.scalar_one_or_none.return_value = datetime.now()
                return result
            result.scalars.return_value.all.return_value = [mock_game]
            return result

        mock_db.execute = mock_exec

        mock_predictor = MagicMock(is_ready=True)
        mock_predictor.predict_upcoming = AsyncMock(return_value=[mock_pred])

        mock_settings = MagicMock()
        mock_settings.teams_team_id = "tid"
        mock_settings.teams_channel_id = "cid"
        mock_settings.teams_webhook_url = ""
        mock_settings.api_base_url = "https://api.example.com"
        mock_settings.teams_max_games_per_message = 10

        with (
            patch(f"{_SCHED}.poll_stats", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_SCHED}.get_settings", return_value=mock_settings),
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.models.features.reset_elo_cache"),
            patch("src.notifications.teams.build_teams_card", return_value={"body": []}),
            patch("src.notifications.teams.build_slate_csv", return_value="csv_data"),
            patch("src.notifications.teams.upload_csv_to_channel", new_callable=AsyncMock, return_value="https://csv.url"),
            patch("src.notifications.teams.send_card_via_graph", new_callable=AsyncMock) as mock_graph,
            patch("src.notifications.teams.build_html_slate", return_value="<html>"),
            patch("src.notifications.teams.send_html_via_graph", new_callable=AsyncMock),
        ):
            await generate_predictions_and_publish()
        mock_graph.assert_awaited_once()

    @pytest.mark.anyio
    async def test_no_delivery_configured(self):
        """No webhook or Graph config → logs skip."""
        from src.data.scheduler import generate_predictions_and_publish

        mock_sf, mock_db = _mock_session()

        mock_pred = MagicMock()
        mock_pred.game_id = 1
        mock_game = MagicMock()
        mock_game.id = 1

        async def mock_exec(stmt, *a, **kw):
            result = MagicMock()
            sql = str(stmt).lower()
            if "count" in sql:
                result.scalar.return_value = 1
                return result
            if "max" in sql:
                result.scalar_one_or_none.return_value = None
                return result
            result.scalars.return_value.all.return_value = [mock_game]
            return result

        mock_db.execute = mock_exec

        mock_predictor = MagicMock(is_ready=True)
        mock_predictor.predict_upcoming = AsyncMock(return_value=[mock_pred])

        mock_settings = MagicMock()
        mock_settings.teams_team_id = ""
        mock_settings.teams_channel_id = ""
        mock_settings.teams_webhook_url = ""
        mock_settings.api_base_url = ""
        mock_settings.teams_max_games_per_message = 10

        with (
            patch(f"{_SCHED}.poll_stats", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_scores_and_box", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_injuries", new_callable=AsyncMock),
            patch(f"{_SCHED}.sync_events_to_games", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_fg_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_1h_odds", new_callable=AsyncMock),
            patch(f"{_SCHED}.poll_player_props", new_callable=AsyncMock),
            patch(f"{_SCHED}.get_settings", return_value=mock_settings),
            patch(f"{_SCHED}.async_session_factory", mock_sf),
            patch("src.models.predictor.Predictor", return_value=mock_predictor),
            patch("src.models.features.reset_elo_cache"),
            patch("src.notifications.teams.build_teams_card", return_value={"body": []}),
        ):
            await generate_predictions_and_publish()


# ── prune_old_odds log ───────────────────────────────────────────


class TestPruneOldOddsLog:
    @pytest.mark.anyio
    async def test_prune_logs_count(self):
        from src.data.scheduler import prune_old_odds

        mock_sf, mock_db = _mock_session()
        result = MagicMock()
        result.rowcount = 100
        mock_db.execute = AsyncMock(return_value=result)
        mock_db.commit = AsyncMock()

        with patch(f"{_SCHED}.async_session_factory", mock_sf):
            await prune_old_odds()


# ── db_maintenance log ───────────────────────────────────────────


class TestDbMaintenanceLog:
    @pytest.mark.anyio
    async def test_maintenance_logs_success(self):
        from src.data.scheduler import db_maintenance

        mock_sf, mock_db = _mock_session()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()

        with patch(f"{_SCHED}.async_session_factory", mock_sf):
            await db_maintenance()
