"""Teams message delivery: webhooks, Graph API, file uploads."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ── Simple alert (webhook) ─────────────────────────────────────────


async def send_alert(title: str, message: str, severity: str = "warning") -> None:
    """Send a lightweight alert to Teams via the configured webhook.

    Designed for critical system alerts (retrain failure, quota exhaustion,
    etc.).  Silently swallows delivery errors to avoid masking the original
    failure that triggered the alert.
    """
    from src.config import get_settings

    settings = get_settings()
    webhook_url = settings.teams_webhook_url
    if not webhook_url:
        logger.debug("No Teams webhook configured; skipping alert")
        return

    color = {"error": "attention", "warning": "warning"}.get(severity, "default")
    card: dict[str, Any] = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"⚠ {title}",
                            "weight": "bolder",
                            "size": "medium",
                            "color": color,
                        },
                        {
                            "type": "TextBlock",
                            "text": message,
                            "wrap": True,
                        },
                        {
                            "type": "TextBlock",
                            "text": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                            "size": "small",
                            "isSubtle": True,
                        },
                    ],
                },
            }
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(webhook_url, json=card)
            resp.raise_for_status()
        logger.info("Alert sent to Teams: %s", title)
    except Exception:
        logger.warning("Failed to send Teams alert: %s", title, exc_info=True)


# ── Payload helpers ────────────────────────────────────────────────


def _payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False))


def _chunk_card_payload(
    payload: dict[str, Any],
    max_payload_bytes: int = 26_000,
) -> list[dict[str, Any]]:
    import copy

    if _payload_size_bytes(payload) <= max_payload_bytes:
        return [payload]

    try:
        original_content = payload["attachments"][0]["content"]
        original_body = list(original_content["body"])
    except (KeyError, IndexError, TypeError):
        return [payload]

    if len(original_body) <= 4:
        return [payload]

    shared_prefix = original_body[:4]
    variable_items = original_body[4:]
    chunks: list[dict[str, Any]] = []

    def _new_chunk() -> tuple[dict[str, Any], dict[str, Any]]:
        chunk_payload = copy.deepcopy(payload)
        chunk_content = chunk_payload["attachments"][0]["content"]
        chunk_content["body"] = list(shared_prefix)
        return chunk_payload, chunk_content

    chunk_payload, chunk_content = _new_chunk()
    variable_items_in_chunk = 0

    for item in variable_items:
        chunk_content["body"].append(item)
        if _payload_size_bytes(chunk_payload) > max_payload_bytes and variable_items_in_chunk > 0:
            chunk_content["body"].pop()
            chunk_content.pop("actions", None)
            chunks.append(chunk_payload)
            chunk_payload, chunk_content = _new_chunk()
            chunk_content["body"].append(item)
            variable_items_in_chunk = 1
            continue

        variable_items_in_chunk += 1

    chunks.append(chunk_payload)
    for chunk in chunks[:-1]:
        chunk["attachments"][0]["content"].pop("actions", None)
    return chunks


# ── Webhook senders ────────────────────────────────────────────────


async def send_card_to_teams(webhook_url: str, payload: dict[str, Any]) -> None:
    """Send an Adaptive Card payload to a Teams incoming webhook.

    Automatically chunks the payload into multiple cards if it exceeds
    Power Automate's 28 KB limit.
    """
    raw_size = _payload_size_bytes(payload)
    chunks = _chunk_card_payload(payload)
    if len(chunks) == 1:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
        return

    logger.info(
        "Payload is %d bytes (exceeds 25KB limit). Chunking Adaptive Card into %d parts...",
        raw_size,
        len(chunks),
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, chunk_payload in enumerate(chunks, start=1):
            response = await client.post(webhook_url, json=chunk_payload)
            response.raise_for_status()
            logger.info("Successfully sent Teams payload chunk %d/%d", i, len(chunks))


async def send_text_to_teams(webhook_url: str, text: str) -> None:
    payload: dict[str, Any] = {"text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(webhook_url, json=payload)
        response.raise_for_status()


# ── Graph API senders ──────────────────────────────────────────────


async def send_card_via_graph(
    team_id: str, channel_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Post an Adaptive Card to a Teams channel via Microsoft Graph API.

    Uses ``DefaultAzureCredential`` to obtain a token (works with
    Azure CLI, managed identity, environment creds, etc.).

    ``payload`` should be the dict returned by ``build_teams_card()``.
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")

    attachment = payload["attachments"][0]
    card_content = attachment["content"]

    graph_body = {
        "body": {
            "contentType": "html",
            "content": '<attachment id="card-1"></attachment>',
        },
        "attachments": [
            {
                "id": "card-1",
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": json.dumps(card_content, ensure_ascii=False),
            }
        ],
    }

    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json=graph_body,
            headers={
                "Authorization": f"Bearer {token.token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()


async def send_html_via_graph(team_id: str, channel_id: str, html_content: str) -> dict[str, Any]:
    """Post an HTML message directly to a Teams channel via Microsoft Graph API.

    Renders as native HTML in Teams — no Adaptive Card, no login prompt.
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")

    graph_body = {
        "body": {
            "contentType": "html",
            "content": html_content,
        },
    }

    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json=graph_body,
            headers={
                "Authorization": f"Bearer {token.token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()


async def upload_csv_to_channel(
    team_id: str,
    channel_id: str,
    filename: str,
    csv_content: str,
) -> str:
    """Upload a CSV file to the Teams channel's Files tab and return the web URL.

    Uses the Graph API to:
    1. Get the channel's filesFolder (SharePoint drive)
    2. PUT the CSV content to that folder
    3. Return the webUrl for the uploaded file
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")
    headers = {
        "Authorization": f"Bearer {token.token}",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        folder_url = (
            f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/filesFolder"
        )
        folder_resp = await client.get(folder_url, headers=headers)
        folder_resp.raise_for_status()
        folder_data = folder_resp.json()

        drive_id = folder_data["parentReference"]["driveId"]
        folder_id = folder_data["id"]

        upload_url = (
            f"https://graph.microsoft.com/v1.0/drives/{drive_id}"
            f"/items/{folder_id}:/{filename}:/content"
        )
        upload_resp = await client.put(
            upload_url,
            content=csv_content.encode("utf-8-sig"),
            headers={
                **headers,
                "Content-Type": "text/csv",
            },
        )
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()

        web_url: str = upload_data.get("webUrl", "")
        logger.info("Uploaded %s → %s", filename, web_url)
        return web_url


async def upload_html_to_channel(
    team_id: str,
    channel_id: str,
    filename: str,
    html_content: str,
) -> str:
    """Upload an HTML file to the Teams channel's Files tab and return the web URL.

    Uses the Graph API to:
    1. Get the channel's filesFolder (SharePoint drive)
    2. PUT the HTML content to that folder
    3. Return the webUrl for the uploaded file (usable as a direct download/OneDrive link)
    """
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://graph.microsoft.com/.default")
    headers = {
        "Authorization": f"Bearer {token.token}",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        folder_url = (
            f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/filesFolder"
        )
        folder_resp = await client.get(folder_url, headers=headers)
        folder_resp.raise_for_status()
        folder_data = folder_resp.json()

        drive_id = folder_data["parentReference"]["driveId"]
        folder_id = folder_data["id"]

        upload_url = (
            f"https://graph.microsoft.com/v1.0/drives/{drive_id}"
            f"/items/{folder_id}:/{filename}:/content"
        )
        upload_resp = await client.put(
            upload_url,
            content=html_content.encode("utf-8"),
            headers={
                **headers,
                "Content-Type": "text/html",
            },
        )
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()

        web_url: str = upload_data.get("webUrl", "")
        logger.info("Uploaded HTML %s → %s", filename, web_url)
        return web_url
