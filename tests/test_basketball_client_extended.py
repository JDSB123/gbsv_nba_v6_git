"""Tests for BasketballClient — _get, _headers, _resolve_season,
normalize_team_stats, _compute_advanced_stats, _pct_to_decimal,
persist_teams, persist_games, persist_team_season_stats,
persist_player_game_stats, persist_injuries, INJURY_STATUS_MAP.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.basketball_client import (
    INJURY_STATUS_MAP,
    BasketballClient,
    _as_float,
    _as_int,
    _compute_advanced_stats,
    _pct_to_decimal,
    normalize_team_stats,
)

# ── Pure helpers ─────────────────────────────────────────────────


class TestAsFloat:
    def test_valid(self):
        assert _as_float("3.14") == 3.14

    def test_none(self):
        assert _as_float(None) is None

    def test_empty_string(self):
        assert _as_float("") is None


class TestAsInt:
    def test_valid(self):
        assert _as_int("42") == 42

    def test_none_default(self):
        assert _as_int(None) == 0

    def test_empty_default(self):
        assert _as_int("", 99) == 99


class TestPctToDecimal:
    def test_whole_number_pct(self):
        # 46.5 (meaning 46.5%) → 0.465
        assert abs(_pct_to_decimal("46.5") - 0.465) < 1e-6

    def test_already_decimal(self):
        # 0.465 is already ≤1, kept as-is
        assert abs(_pct_to_decimal("0.465") - 0.465) < 1e-6

    def test_none_returns_none(self):
        assert _pct_to_decimal(None) is None

    def test_zero_returns_zero(self):
        assert _pct_to_decimal("0") == 0.0

    def test_negative_returns_none(self):
        assert _pct_to_decimal("-5") is None


class TestNormalizeTeamStats:
    def test_dict_passthrough(self):
        d = {"games": {}}
        assert normalize_team_stats(d) is d

    def test_list_of_dicts(self):
        lst = [{"games": {}}, {"extra": True}]
        assert normalize_team_stats(lst) == {"games": {}}

    def test_none_input(self):
        assert normalize_team_stats(None) is None

    def test_empty_list(self):
        assert normalize_team_stats([]) is None


class TestComputeAdvancedStats:
    def test_primary_path_with_full_stats(self):
        s = {
            "field_goals": {"total": {"all": "800"}, "percentage": {"all": "46.5"}},
            "free_throws": {"total": {"all": "200"}, "percentage": {"all": "80.0"}},
            "rebounds": {"offReb": {"all": "100"}},
            "turnovers": {"total": {"all": "150"}},
            "points": {
                "for": {"total": {"all": "2000"}},
                "against": {"total": {"all": "1900"}},
            },
        }
        pace, off_r, def_r = _compute_advanced_stats(s, games_played=40, ppg=100.0, oppg=95.0)
        assert pace is not None
        assert off_r is not None
        assert def_r is not None
        assert pace > 0

    def test_fallback_to_ppg_oppg(self):
        s = {}
        pace, off_r, def_r = _compute_advanced_stats(s, games_played=40, ppg=112.0, oppg=108.0)
        assert pace is not None
        assert off_r is not None
        assert def_r is not None

    def test_no_games_played(self):
        s = {}
        pace, off_r, def_r = _compute_advanced_stats(s, games_played=0, ppg=None, oppg=None)
        assert pace is None
        assert off_r is None
        assert def_r is None


class TestInjuryStatusMap:
    def test_known_statuses(self):
        assert INJURY_STATUS_MAP["out"] == "out"
        assert INJURY_STATUS_MAP["out for season"] == "out"
        assert INJURY_STATUS_MAP["doubtful"] == "doubtful"
        assert INJURY_STATUS_MAP["day-to-day"] == "questionable"
        assert INJURY_STATUS_MAP["questionable"] == "questionable"
        assert INJURY_STATUS_MAP["probable"] == "probable"


# ── Client construction ──────────────────────────────────────────


class TestClientConstruction:
    def test_headers(self):
        with patch("src.data.basketball_client.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                basketball_api_base="https://api.example.com",
                basketball_api_key="test-key-123",
                basketball_api_league_id=12,
                nba_api_base="https://nba.example.com",
            )
            client = BasketballClient()
            headers = client._headers()
            assert headers["x-apisports-key"] == "test-key-123"

    def test_resolve_season_with_value(self):
        with patch("src.data.basketball_client.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                basketball_api_base="https://api.example.com",
                basketball_api_key="key",
                basketball_api_league_id=12,
                nba_api_base="https://nba.example.com",
            )
            client = BasketballClient()
            assert client._resolve_season("2024-2025") == "2024-2025"

    def test_resolve_season_none_uses_current(self):
        with patch("src.data.basketball_client.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                basketball_api_base="https://api.example.com",
                basketball_api_key="key",
                basketball_api_league_id=12,
                nba_api_base="https://nba.example.com",
            )
            client = BasketballClient()
            result = client._resolve_season(None)
            assert isinstance(result, str)
            assert "-" in result  # e.g. "2024-2025"


# ── _get ─────────────────────────────────────────────────────────


class TestGet:
    @patch("httpx.AsyncClient")
    async def test_get_returns_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": [{"id": 1}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.basketball_client.get_settings") as mock_settings:
            mock_settings.return_value = SimpleNamespace(
                basketball_api_base="https://api.example.com",
                basketball_api_key="key",
                basketball_api_league_id=12,
                nba_api_base="https://nba.example.com",
            )
            client = BasketballClient()
            result = await client._get("games", {"date": "2025-01-15"})
            assert result == [{"id": 1}]
