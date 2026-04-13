"""Tests for src.data.health_check — API health probes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.health_check import (
    _last_check_results,
    _last_check_time,
    check_basketball_api,
    check_database,
    check_odds_api,
    get_last_check,
    run_startup_health_check,
)


@pytest.mark.asyncio
class TestCheckOddsApi:
    async def test_ok_when_nba_found(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"key": "basketball_nba"}]
        mock_resp.headers = {"x-requests-remaining": "450", "x-requests-used": "50"}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.health_check.get_settings") as ms:
            ms.return_value.odds_api_base = "https://api.test"
            ms.return_value.odds_api_key = "testkey"
            ms.return_value.odds_api_sport_key = "basketball_nba"
            ms.return_value.odds_api_regions = "us,us2"
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
                result = await check_odds_api()
                assert result["status"] == "ok"
                assert result["nba_active"] is True
                assert result["quota_remaining"] == 450

    async def test_warning_when_nba_missing(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"key": "soccer_epl"}]
        mock_resp.headers = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.health_check.get_settings") as ms:
            ms.return_value.odds_api_base = "https://api.test"
            ms.return_value.odds_api_key = "key"
            ms.return_value.odds_api_sport_key = "basketball_nba"
            ms.return_value.odds_api_regions = "us"
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
                result = await check_odds_api()
                assert result["status"] == "warning"

    async def test_error_on_exception(self):
        with patch("src.data.health_check.get_settings") as ms:
            ms.return_value.odds_api_base = "https://api.test"
            ms.return_value.odds_api_key = "key"
            ms.return_value.odds_api_sport_key = "basketball_nba"
            with patch("httpx.AsyncClient.get", side_effect=Exception("timeout")):
                result = await check_odds_api()
                assert result["status"] == "error"


@pytest.mark.asyncio
class TestCheckBasketballApi:
    async def test_ok(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": {
                "account": {"subscription": {"plan": "Mega"}},
                "requests": {"current": 100, "limit_day": 10000},
            }
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data.health_check.get_settings") as ms:
            ms.return_value.basketball_api_base = "https://api.test"
            ms.return_value.basketball_api_key = "key"
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
                result = await check_basketball_api()
                assert result["status"] == "ok"
                assert result["plan"] == "Mega"
                assert result["requests_today"] == 100

    async def test_error(self):
        with patch("src.data.health_check.get_settings") as ms:
            ms.return_value.basketball_api_base = "https://api.test"
            ms.return_value.basketball_api_key = "key"
            with patch("httpx.AsyncClient.get", side_effect=Exception("refused")):
                result = await check_basketball_api()
                assert result["status"] == "error"


@pytest.mark.asyncio
class TestCheckDatabase:
    async def test_ok(self):
        mock_session = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("src.db.session.async_session_factory", return_value=mock_cm):
            result = await check_database()
            assert result["status"] == "ok"

    async def test_error(self):
        with patch("src.db.session.async_session_factory", side_effect=Exception("no db")):
            result = await check_database()
            assert result["status"] == "error"


@pytest.mark.asyncio
class TestRunStartupHealthCheck:
    async def test_runs_all_checks(self):
        with (
            patch("src.data.health_check.check_database", new_callable=AsyncMock) as mock_db,
            patch("src.data.health_check.check_odds_api", new_callable=AsyncMock) as mock_odds,
            patch(
                "src.data.health_check.check_basketball_api", new_callable=AsyncMock
            ) as mock_bball,
        ):
            mock_db.return_value = {"source": "database", "status": "ok"}
            mock_odds.return_value = {
                "source": "odds_api_v4",
                "status": "ok",
                "quota_remaining": 500,
                "regions": "us",
            }
            mock_bball.return_value = {
                "source": "basketball_api_v1",
                "status": "ok",
                "plan": "Mega",
                "pct_used": 1.0,
            }
            results = await run_startup_health_check()
            assert len(results) == 3
            assert all(r["status"] == "ok" for r in results)


class TestGetLastCheck:
    def test_returns_cached_results(self):
        _last_check_results.clear()
        _last_check_results.append({"source": "test", "status": "ok"})
        _last_check_time[0] = "2025-01-01"
        results, ts = get_last_check()
        assert len(results) == 1
        assert ts == "2025-01-01"
        _last_check_results.clear()
        _last_check_time[0] = None
