"""Tests for src.notifications._delivery — payload helpers and webhook delivery."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.notifications._delivery import (
    _chunk_card_payload,
    _payload_size_bytes,
    send_alert,
    send_card_to_teams,
    send_text_to_teams,
)


class TestPayloadSizeBytes:
    def test_simple(self):
        payload = {"text": "hello"}
        assert _payload_size_bytes(payload) == len(json.dumps(payload, ensure_ascii=False))

    def test_empty(self):
        assert _payload_size_bytes({}) == 2  # "{}"

    def test_unicode(self):
        payload = {"emoji": "\U0001f525"}
        size = _payload_size_bytes(payload)
        assert size > 0


class TestChunkCardPayload:
    def _make_large_payload(self, n_items: int = 50):
        body = [{"type": "TextBlock", "text": f"Header {i}"} for i in range(4)]
        body += [{"type": "TextBlock", "text": "X" * 600} for _ in range(n_items)]
        return {
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": body,
                        "actions": [{"type": "Action.OpenUrl", "title": "View"}],
                    },
                }
            ]
        }

    def test_small_payload_single_chunk(self):
        payload = {"attachments": [{"content": {"body": [{"text": "hi"}]}}]}
        chunks = _chunk_card_payload(payload, max_payload_bytes=100_000)
        assert len(chunks) == 1

    def test_large_payload_splits(self):
        payload = self._make_large_payload(50)
        chunks = _chunk_card_payload(payload, max_payload_bytes=5_000)
        assert len(chunks) > 1

    def test_all_items_preserved(self):
        payload = self._make_large_payload(20)
        n_original = len(payload["attachments"][0]["content"]["body"])
        chunks = _chunk_card_payload(payload, max_payload_bytes=5_000)
        total_items = sum(len(c["attachments"][0]["content"]["body"]) for c in chunks)
        # Each chunk includes 4 shared prefix items, so total >= original
        assert total_items >= n_original

    def test_malformed_payload_returns_as_is(self):
        payload = {"bad": "structure"}
        chunks = _chunk_card_payload(payload, max_payload_bytes=1)
        assert len(chunks) == 1
        assert chunks[0] == payload

    def test_small_body_returns_as_is(self):
        """Payload with <=4 body items isn't worth splitting."""
        payload = {
            "attachments": [
                {"content": {"body": [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}]}}
            ]
        }
        chunks = _chunk_card_payload(payload, max_payload_bytes=1)
        assert len(chunks) == 1


@pytest.mark.asyncio
class TestSendAlert:
    async def test_no_webhook_configured(self):
        """Should silently return when no webhook URL is set."""
        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.teams_webhook_url = ""
            await send_alert("Test", "Some message")  # Should not raise

    async def test_sends_to_webhook(self):
        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.teams_webhook_url = "https://webhook.example.com/test"
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value.raise_for_status = lambda: None
                await send_alert("Alert Title", "Alert body", severity="error")
                mock_post.assert_awaited_once()
                call_kwargs = mock_post.call_args
                assert "https://webhook.example.com/test" in str(call_kwargs)

    async def test_swallows_errors(self):
        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.teams_webhook_url = "https://webhook.example.com/test"
            with patch(
                "httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("fail")
            ):
                await send_alert("Title", "body")  # Should not raise


@pytest.mark.asyncio
class TestSendTextToTeams:
    async def test_sends_text(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.raise_for_status = lambda: None
            await send_text_to_teams("https://webhook.example.com", "Hello!")
            mock_post.assert_awaited_once()
            payload = mock_post.call_args[1]["json"]
            assert payload["text"] == "Hello!"


@pytest.mark.asyncio
class TestSendCardToTeams:
    async def test_sends_single_chunk(self):
        payload = {"attachments": [{"content": {"body": [{"text": "small"}]}}]}
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.raise_for_status = lambda: None
            await send_card_to_teams("https://webhook.example.com", payload)
            assert mock_post.await_count == 1
