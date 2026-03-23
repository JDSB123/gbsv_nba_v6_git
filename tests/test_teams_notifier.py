from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.versioning import MODEL_VERSION
from src.notifications.teams import (
    build_slate_csv,
    build_teams_card,
    build_teams_text,
    extract_picks,
    send_card_via_graph,
)


def test_build_teams_text_formats_matchups():
    game = SimpleNamespace(
        id=1,
        home_team_id=10,
        away_team_id=20,
        commence_time=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        home_team=SimpleNamespace(name="Boston Celtics"),
        away_team=SimpleNamespace(name="Miami Heat"),
    )
    pred = SimpleNamespace(
        game_id=1,
        fg_spread=3.5,
        fg_total=224.5,
        fg_home_ml_prob=0.612,
    )

    text = build_teams_text([(pred, game)], max_games=5)

    assert "NBA Predictions Update" in text
    assert "Miami Heat at Boston Celtics" in text
    assert "FG spread +3.5" in text
    assert "FG total 224.5" in text
    assert "FG ML H 0.612 / A 0.388" in text


def test_build_teams_text_honors_max_games():
    game = SimpleNamespace(
        id=1,
        home_team_id=10,
        away_team_id=20,
        commence_time=None,
        home_team=SimpleNamespace(name="A"),
        away_team=SimpleNamespace(name="B"),
    )
    pred = SimpleNamespace(
        game_id=1,
        fg_spread=0.0,
        fg_total=210.0,
        fg_home_ml_prob=0.5,
    )

    text = build_teams_text([(pred, game), (pred, game)], max_games=1)

    assert "Games: 1" in text
    assert text.count(" at ") == 1


def test_build_teams_card_has_adaptive_card_structure():
    game = SimpleNamespace(
        id=1,
        home_team_id=10,
        away_team_id=20,
        commence_time=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        home_team=SimpleNamespace(name="Boston Celtics", wins=42, losses=18),
        away_team=SimpleNamespace(name="Miami Heat", wins=30, losses=30),
    )
    pred = SimpleNamespace(
        game_id=1,
        fg_spread=6.5,
        fg_total=224.5,
        fg_home_ml_prob=0.72,
        h1_spread=3.5,
        h1_total=112.0,
        opening_spread=None,
        predicted_home_fg=115.5,
        predicted_away_fg=109.0,
        predicted_home_1h=57.8,
        predicted_away_1h=54.2,
    )

    odds_ts = datetime(2026, 3, 22, 18, 0, tzinfo=UTC)
    payload = build_teams_card([(pred, game)], max_games=5, odds_pulled_at=odds_ts)

    assert payload["type"] == "message"
    assert len(payload["attachments"]) == 1
    card = payload["attachments"][0]["content"]
    assert card["type"] == "AdaptiveCard"
    assert card["version"] == "1.4"

    import json

    card_str = json.dumps(card)
    assert MODEL_VERSION in card_str
    assert "Odds pulled" in card_str
    assert "2026-03-22 18:00 UTC" in card_str
    # Pick-based format markers
    assert "PICK" in card_str
    assert "EDGE" in card_str
    assert "qualified picks" in card_str
    # New: segment and detail line markers
    assert "FG SPREAD" in card_str or "FG TOTAL" in card_str or "FG ML" in card_str
    assert "Model:" in card_str
    # Records should appear
    assert "42-18" in card_str
    assert "30-30" in card_str


def test_build_teams_card_sorts_by_edge():
    def base_game(tid, name):
        return SimpleNamespace(
            id=tid,
            home_team_id=tid,
            away_team_id=tid + 100,
            commence_time=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
            home_team=SimpleNamespace(name=name, wins=None, losses=None),
            away_team=SimpleNamespace(name="Visitor", wins=None, losses=None),
        )

    high_pred = SimpleNamespace(
        game_id=1,
        fg_spread=10.0,
        fg_total=235.0,
        fg_home_ml_prob=0.80,
        h1_spread=5.0,
        h1_total=117.0,
        predicted_home_fg=122.5,
        predicted_away_fg=112.5,
        predicted_home_1h=61.0,
        predicted_away_1h=56.0,
    )
    low_pred = SimpleNamespace(
        game_id=2,
        fg_spread=2.5,
        fg_total=220.0,
        fg_home_ml_prob=0.55,
        h1_spread=1.0,
        h1_total=110.0,
        predicted_home_fg=111.3,
        predicted_away_fg=108.7,
        predicted_home_1h=55.5,
        predicted_away_1h=54.5,
    )
    rows = [
        (low_pred, base_game(2, "Team Low")),
        (high_pred, base_game(1, "Team High")),
    ]
    payload = build_teams_card(rows, max_games=10)
    card = payload["attachments"][0]["content"]

    import json

    card_str = json.dumps(card)
    # High-edge pick (Team High has fg_spread=10) should appear before low
    high_pos = card_str.index("Team High")
    low_pos = card_str.index("Team Low")
    assert high_pos < low_pos, "Higher-edge pick should appear first"

    # Summary should show game and pick counts
    assert "2 games" in card_str
    assert "qualified picks" in card_str


def test_extract_picks_produces_spread_total_ml():
    game = SimpleNamespace(
        id=1,
        home_team_id=10,
        away_team_id=20,
        commence_time=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        home_team=SimpleNamespace(name="Celtics", wins=42, losses=18),
        away_team=SimpleNamespace(name="Heat", wins=30, losses=30),
    )
    pred = SimpleNamespace(
        game_id=1,
        fg_spread=8.0,
        fg_total=232.0,
        fg_home_ml_prob=0.75,
        h1_spread=4.0,
        h1_total=116.0,
        opening_spread=None,
        predicted_home_fg=120.0,
        predicted_away_fg=112.0,
        predicted_home_1h=60.0,
        predicted_away_1h=56.0,
    )
    picks = extract_picks(pred, game)

    labels = [p.label for p in picks]
    # Should have spread, total, and ML picks
    assert any("Celtics" in label for label in labels)  # home favorite
    assert any("OVER" in label or "UNDER" in label for label in labels)  # total pick
    assert any("ML" in label for label in labels)  # ML pick
    # 1H picks use segment field, not label suffix
    assert any(p.segment == "1H" for p in picks)
    # All edges should be positive
    assert all(p.edge > 0 for p in picks)
    # New Pick fields should be populated
    segments = {p.segment for p in picks}
    assert "FG" in segments
    market_types = {p.market_type for p in picks}
    assert len(market_types) > 1
    assert all(p.home_record == "42-18" for p in picks)
    assert all(p.away_record == "30-30" for p in picks)
    assert all(1 <= p.confidence <= 5 for p in picks)
    # Fire emojis should appear in card pick rows
    payload = build_teams_card([(pred, game)], max_games=5)
    card = payload["attachments"][0]["content"]
    pick_texts = [
        item["text"]
        for elem in card["body"]
        if elem.get("type") == "ColumnSet" and "columns" in elem
        for col in elem["columns"]
        for item in col.get("items", [])
        if "🔥" in item.get("text", "")
    ]
    assert len(pick_texts) > 0, "Fire emojis should appear in pick labels"


def test_build_slate_csv():
    game = SimpleNamespace(
        id=1,
        home_team_id=10,
        away_team_id=20,
        commence_time=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        home_team=SimpleNamespace(name="Celtics", wins=42, losses=18),
        away_team=SimpleNamespace(name="Heat", wins=30, losses=30),
    )
    pred = SimpleNamespace(
        game_id=1,
        fg_spread=8.0,
        fg_total=232.0,
        fg_home_ml_prob=0.75,
        h1_spread=4.0,
        h1_total=116.0,
        opening_spread=None,
        predicted_home_fg=120.0,
        predicted_away_fg=112.0,
        predicted_home_1h=60.0,
        predicted_away_1h=56.0,
    )
    csv_text = build_slate_csv([(pred, game)])
    lines = csv_text.strip().split("\n")
    # Header + at least one data row
    assert len(lines) >= 2
    header = lines[0]
    assert "Time (CT)" in header
    assert "Matchup" in header
    assert "Edge" in header
    assert "Rating" in header
    # Data rows should contain team names
    data = "\n".join(lines[1:])
    assert "Celtics" in data or "Heat" in data


@pytest.mark.asyncio
async def test_send_card_via_graph_posts_correct_format():
    """Verify Graph API sender rewrites the payload correctly."""
    game = SimpleNamespace(
        id=1,
        home_team_id=10,
        away_team_id=20,
        commence_time=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        home_team=SimpleNamespace(name="Celtics", wins=42, losses=18),
        away_team=SimpleNamespace(name="Heat", wins=30, losses=30),
    )
    pred = SimpleNamespace(
        game_id=1,
        fg_spread=6.5,
        fg_total=224.5,
        fg_home_ml_prob=0.70,
        h1_spread=3.5,
        h1_total=112.0,
        predicted_home_fg=115.5,
        predicted_away_fg=109.0,
        predicted_home_1h=57.8,
        predicted_away_1h=54.2,
    )
    payload = build_teams_card([(pred, game)], max_games=5)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"id": "msg-123"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_token = MagicMock()
    mock_token.token = "fake-token"
    mock_credential = MagicMock()
    mock_credential.get_token.return_value = mock_token

    with (
        patch("src.notifications.teams.httpx.AsyncClient", return_value=mock_client),
        patch("azure.identity.DefaultAzureCredential", return_value=mock_credential),
    ):
        result = await send_card_via_graph("team-id", "channel-id", payload)

    assert result == {"id": "msg-123"}
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    posted_body = call_kwargs.kwargs["json"]
    # Graph format has body.contentType = "html" with attachment reference
    assert posted_body["body"]["contentType"] == "html"
    assert '<attachment id="card-1">' in posted_body["body"]["content"]
    # Card content should be a JSON string, not a dict
    assert isinstance(posted_body["attachments"][0]["content"], str)
    assert "Authorization" in call_kwargs.kwargs["headers"]
