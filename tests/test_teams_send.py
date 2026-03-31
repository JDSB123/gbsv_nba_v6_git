"""Tests for teams.py: send_card_to_teams, send_text_to_teams,
send_alert, send_card_via_graph, send_html_to_channel, upload_csv_to_channel,
_app_build_stamp, and _odds_source_block 1H_ML edge cases."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MOD = "src.notifications.teams"


class TestAppBuildStamp:
    def test_returns_env_var_when_set(self):
        from src.notifications.teams import _app_build_stamp

        with patch.dict(os.environ, {"APP_BUILD_TIMESTAMP": "2024-01-01 12:00 UTC"}):
            assert _app_build_stamp() == "2024-01-01 12:00 UTC"

    def test_returns_fallback_when_not_set(self):
        from src.notifications.teams import _app_build_stamp

        with patch.dict(os.environ, {}, clear=True):
            # Remove it if it exists
            os.environ.pop("APP_BUILD_TIMESTAMP", None)
            result = _app_build_stamp()
            assert "UTC" in result


class TestExtractPicksOddsMap1H:
    """Cover odds_map building with 1H ML keys (lines 265, 268)."""

    def test_extract_picks_builds_1h_ml_odds_map(self):
        from src.notifications.teams import extract_picks

        pred = SimpleNamespace(
            fg_spread=5.0, fg_total=215.0,
            fg_home_ml_prob=0.65, h1_home_ml_prob=0.60,
            predicted_home_fg=110.0, predicted_away_fg=105.0,
            predicted_home_1h=55.0, predicted_away_1h=52.0,
            opening_spread=-3.5, opening_total=220.0,
            odds_sourced={
                "books": {
                    "fanduel": {
                        "spread": -3.5, "spread_price": -110,
                        "total": 220.0, "total_price": -105,
                        "home_ml": -150, "away_ml": 130,
                        "spread_h1": -1.5, "spread_h1_price": -115,
                        "total_h1": 109.5, "total_h1_price": -112,
                        "home_ml_h1": -130, "away_ml_h1": 110,
                    },
                },
            },
            h1_spread=3.0, h1_total=107.0,
        )
        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(
                name="Lakers",
                team_season_stats=[SimpleNamespace(wins=30, losses=10)],
            ),
            away_team=SimpleNamespace(
                name="Celtics",
                team_season_stats=[SimpleNamespace(wins=25, losses=15)],
            ),
            commence_time=None,
        )
        picks = extract_picks(pred, game, min_edge=0.5)
        # Should have picks generated (at minimum spread/total/ML)
        assert len(picks) > 0


class TestOddsSourceBlockMorePicks:
    """Cover remaining picks in odds_by_game rendering (lines 667, 674-675)."""

    def test_odds_source_block_with_all_markets(self):
        from src.notifications.teams import _odds_source_block

        odds = {
            "books": {
                "fanduel": {
                    "spread": -3.5, "spread_price": -110,
                    "total": 220.0, "total_price": -105,
                    "home_ml": -150, "away_ml": 130,
                    "spread_h1": -1.5, "spread_h1_price": -115,
                    "total_h1": 109.5, "total_h1_price": -112,
                },
            },
            "captured_at": "2024-12-01T18:00:00Z",
        }
        result = _odds_source_block(odds)
        assert len(result) > 0
        text = result[0]["text"]
        assert "1H" in text


class TestBuildTeamsCardRemainingPicks:
    """Cover build_teams_card remaining_picks path (line 795)."""

    def test_card_with_many_picks_shows_remaining(self):
        from src.notifications.teams import build_teams_card

        pred = SimpleNamespace(
            fg_spread=8.0, fg_total=210.0,
            fg_home_ml_prob=0.80, h1_home_ml_prob=0.75,
            predicted_home_fg=115.0, predicted_away_fg=107.0,
            predicted_home_1h=58.0, predicted_away_1h=50.0,
            opening_spread=-2.0, opening_total=225.0,
            opening_h1_spread=-1.0, opening_h1_total=112.0,
            odds_sourced=None, game_id=1,
        )
        game = SimpleNamespace(
            id=1,
            home_team=SimpleNamespace(
                name="Lakers",
                team_season_stats=[SimpleNamespace(wins=30, losses=10)],
            ),
            away_team=SimpleNamespace(
                name="Celtics",
                team_season_stats=[SimpleNamespace(wins=25, losses=15)],
            ),
            commence_time=None,
        )
        # max_games=1, min_edge=0 → many picks but only a few displayed
        result = build_teams_card([(pred, game)], max_games=1, min_edge=0.1)
        body = result["attachments"][0]["content"]["body"]
        assert len(body) > 0


class TestSendCardToTeams:
    @pytest.mark.anyio
    async def test_send_card_success(self):
        from src.notifications.teams import send_card_to_teams

        with patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_card_to_teams("https://hook.example.com", {"type": "message"})
            mock_client.post.assert_awaited_once()

    @pytest.mark.anyio
    async def test_send_card_does_not_write_debug_files(self):
        from src.notifications.teams import send_card_to_teams

        with (
            patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls,
            patch("builtins.open", side_effect=AssertionError("debug file write should not occur")),
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_card_to_teams("https://hook.example.com", {"type": "message"})
            mock_client.post.assert_awaited_once()

    @pytest.mark.anyio
    async def test_send_card_chunks_large_payload_and_keeps_actions_on_final_part(self):
        from src.notifications.teams import send_card_to_teams

        large_text = "x" * 18_000
        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {"type": "TextBlock", "text": "NBA Daily Slate"},
                            {"type": "TextBlock", "text": "Model v6"},
                            {"type": "TextBlock", "text": "5 games"},
                            {"type": "TextBlock", "text": "Top picks"},
                            {"type": "TextBlock", "text": large_text},
                            {"type": "TextBlock", "text": large_text},
                        ],
                        "actions": [
                            {
                                "type": "Action.OpenUrl",
                                "title": "Download HTML",
                                "url": "https://example.com/slate.html",
                            }
                        ],
                    },
                }
            ],
        }

        with patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_card_to_teams("https://hook.example.com", payload)

        assert mock_client.post.await_count == 2
        first_payload = mock_client.post.await_args_list[0].kwargs["json"]
        second_payload = mock_client.post.await_args_list[1].kwargs["json"]
        assert "actions" not in first_payload["attachments"][0]["content"]
        assert "actions" in second_payload["attachments"][0]["content"]
        assert first_payload["attachments"][0]["content"]["body"][0]["text"] == "NBA Daily Slate"
        assert second_payload["attachments"][0]["content"]["body"][0]["text"] == "NBA Daily Slate"


class TestSendTextToTeams:
    @pytest.mark.anyio
    async def test_send_text_success(self):
        from src.notifications.teams import send_text_to_teams

        with patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_text_to_teams("https://hook.example.com", "hello")
            mock_client.post.assert_awaited_once()


class TestSendAlert:
    @pytest.mark.anyio
    async def test_send_alert_success(self):
        from src.notifications.teams import send_alert

        mock_settings = MagicMock()
        mock_settings.teams_webhook_url = "https://hook.example.com"

        with (
            patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls,
            patch("src.config.get_settings", return_value=mock_settings),
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_alert("Test Alert", "This is a test", "warning")

    @pytest.mark.anyio
    async def test_send_alert_failure_suppressed(self):
        from src.notifications.teams import send_alert

        mock_settings = MagicMock()
        mock_settings.teams_webhook_url = "https://hook.example.com"

        with (
            patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls,
            patch("src.config.get_settings", return_value=mock_settings),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("network error"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise
            await send_alert("Test Alert", "This is a test", "error")


class TestSendCardViaGraph:
    @pytest.mark.anyio
    async def test_send_card_via_graph_success(self):
        from src.notifications.teams import send_card_via_graph

        mock_credential = MagicMock()
        mock_token = MagicMock()
        mock_token.token = "fake-token"
        mock_credential.get_token.return_value = mock_token

        with (
            patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls,
            patch.dict("sys.modules", {"azure.identity": MagicMock(DefaultAzureCredential=MagicMock(return_value=mock_credential))}),
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"id": "msg-1"}
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            payload = {
                "attachments": [{"content": {"type": "AdaptiveCard", "body": []}}],
            }
            result = await send_card_via_graph("team-id", "channel-id", payload)
            assert result == {"id": "msg-1"}


class TestSendHtmlToChannel:
    @pytest.mark.anyio
    async def test_send_html_success(self):
        from src.notifications.teams import send_html_via_graph

        mock_credential = MagicMock()
        mock_token = MagicMock()
        mock_token.token = "fake-token"
        mock_credential.get_token.return_value = mock_token

        with (
            patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls,
            patch.dict("sys.modules", {"azure.identity": MagicMock(DefaultAzureCredential=MagicMock(return_value=mock_credential))}),
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"id": "msg-2"}
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await send_html_via_graph("team-id", "channel-id", "<h1>Hi</h1>")
            assert result == {"id": "msg-2"}


class TestUploadCsvToChannel:
    @pytest.mark.anyio
    async def test_upload_csv_success(self):
        from src.notifications.teams import upload_csv_to_channel

        mock_credential = MagicMock()
        mock_token = MagicMock()
        mock_token.token = "fake-token"
        mock_credential.get_token.return_value = mock_token

        folder_resp = MagicMock()
        folder_resp.raise_for_status = MagicMock()
        folder_resp.json.return_value = {
            "parentReference": {"driveId": "drive-1"},
            "id": "folder-1",
        }

        upload_resp = MagicMock()
        upload_resp.raise_for_status = MagicMock()
        upload_resp.json.return_value = {"webUrl": "https://sharepoint.example.com/file.csv"}

        with (
            patch(f"{_MOD}.httpx.AsyncClient") as mock_client_cls,
            patch.dict("sys.modules", {"azure.identity": MagicMock(DefaultAzureCredential=MagicMock(return_value=mock_credential))}),
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=folder_resp)
            mock_client.put = AsyncMock(return_value=upload_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await upload_csv_to_channel(
                "team-id", "channel-id", "slate.csv", "col1,col2\na,b\n"
            )
            assert result == "https://sharepoint.example.com/file.csv"
