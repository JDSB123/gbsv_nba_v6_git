"""Tests for OddsClient – quota tracking, skip logic, persist_odds."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.odds_client import OddsClient

# ── Construction & quota ───────────────────────────────────────


def test_odds_client_init():
    client = OddsClient()
    assert client.base_url is not None
    assert client.sport == "basketball_nba"
    assert client.quota_remaining is None


def test_track_quota():
    client = OddsClient()
    resp = MagicMock()
    resp.headers = {"x-requests-remaining": "450"}
    client._track_quota(resp)
    assert client.quota_remaining == 450


def test_track_quota_missing_header():
    client = OddsClient()
    resp = MagicMock()
    resp.headers = {}
    client._track_quota(resp)
    assert client.quota_remaining is None


# ── Skip logic ─────────────────────────────────────────────────


def test_should_skip_when_quota_above_min():
    client = OddsClient()
    client._remaining_quota = 100
    assert not client._should_skip()


def test_should_skip_when_quota_below_min():
    client = OddsClient()
    client._remaining_quota = 1  # below default min of 10
    assert client._should_skip()


def test_should_skip_when_quota_unknown():
    client = OddsClient()
    client._remaining_quota = None
    assert not client._should_skip()


def test_should_skip_exactly_at_min():
    client = OddsClient()
    client._remaining_quota = client._quota_min
    assert not client._should_skip()


def test_fetch_odds_returns_empty_when_skip(monkeypatch):
    """When quota is exhausted, fetch_odds returns [] without HTTP call."""
    client = OddsClient()
    client._remaining_quota = 0

    import asyncio

    result = asyncio.run(client.fetch_odds())
    assert result == []


def test_fetch_scores_returns_empty_when_skip():
    """When quota is exhausted, fetch_scores returns [] without HTTP call."""
    client = OddsClient()
    client._remaining_quota = 0

    import asyncio

    result = asyncio.run(client.fetch_scores())
    assert result == []


def test_fetch_event_odds_returns_empty_when_skip():
    client = OddsClient()
    client._remaining_quota = 0

    import asyncio

    result = asyncio.run(client.fetch_event_odds("ev123"))
    assert result == {}


def test_fetch_player_props_returns_empty_when_skip():
    client = OddsClient()
    client._remaining_quota = 0

    import asyncio

    result = asyncio.run(client.fetch_player_props("ev123"))
    assert result == {}


# ── persist_odds ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_odds_inserts_snapshots():
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()

    # Mock: game lookup returns game_id=42
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=42)))

    odds_data = [
        {
            "id": "oa_id_1",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -110, "point": -5.5},
                                {"name": "LA Lakers", "price": -110, "point": 5.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    count = await client.persist_odds(odds_data, db)

    assert count == 2
    assert db.add.call_count == 2
    db.commit.assert_awaited_once()
    assert db.execute.await_count == 2

    # Verify first snapshot
    snap = db.add.call_args_list[0][0][0]
    assert snap.game_id == 42
    assert snap.bookmaker == "fanduel"
    assert snap.market == "spreads"
    assert snap.outcome_name == "Boston Celtics"
    assert snap.price == -110
    assert snap.point == -5.5


@pytest.mark.asyncio
async def test_persist_odds_skips_unknown_game():
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()

    # Game lookup returns None
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))

    odds_data = [
        {
            "id": "unknown_id",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "A", "price": 150}]}],
                }
            ],
        }
    ]

    count = await client.persist_odds(odds_data, db)

    assert count == 0
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_persist_odds_multiple_events():
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()

    # Two events, first found (id=1), second not found
    db.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one_or_none=MagicMock(return_value=1)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
            MagicMock(),
        ]
    )

    odds_data = [
        {
            "id": "ev1",
            "bookmakers": [
                {
                    "key": "dk",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "A", "price": -150}]}],
                }
            ],
        },
        {
            "id": "ev2",
            "bookmakers": [
                {
                    "key": "dk",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "B", "price": 120}]}],
                }
            ],
        },
    ]

    count = await client.persist_odds(odds_data, db)
    assert count == 1
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_odds_multiple_markets_and_books():
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=10)))

    odds_data = [
        {
            "id": "ev1",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {"key": "h2h", "outcomes": [{"name": "Home", "price": -130}]},
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Home", "price": -110, "point": -3.5},
                                {"name": "Away", "price": -110, "point": 3.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 220.5},
                                {"name": "Under", "price": -110, "point": 220.5},
                            ],
                        },
                    ],
                },
            ],
        }
    ]

    count = await client.persist_odds(odds_data, db)
    # fanduel h2h: 1, fanduel spreads: 2, dk totals: 2 = 5
    assert count == 5


@pytest.mark.asyncio
async def test_persist_odds_empty_data():
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()

    count = await client.persist_odds([], db)
    assert count == 0
    db.add.assert_not_called()
    db.commit.assert_awaited_once()


# ── fetch method happy paths (mock httpx) ──────────────────────


def _mock_httpx_response(json_data):
    """Build a mock httpx response with quota header."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.headers = {"x-requests-remaining": "99"}
    return resp


def _make_client():
    """Create an OddsClient with high quota so _should_skip is False."""
    client = OddsClient()
    client._remaining_quota = 500
    return client


@pytest.mark.asyncio
async def test_fetch_events_success():
    client = _make_client()
    expected = [{"id": "evt1"}]
    mock_resp = _mock_httpx_response(expected)
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.data.odds_client.httpx.AsyncClient", lambda: mock_http)
        result = await client.fetch_events()
    assert result == expected
    assert client.quota_remaining == 99


@pytest.mark.asyncio
async def test_fetch_odds_success():
    client = _make_client()
    expected = [{"id": "x", "bookmakers": []}]
    mock_resp = _mock_httpx_response(expected)
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.data.odds_client.httpx.AsyncClient", lambda: mock_http)
        result = await client.fetch_odds()
    assert result == expected


@pytest.mark.asyncio
async def test_fetch_event_odds_success():
    client = _make_client()
    expected = {"id": "ev1", "bookmakers": [{"key": "dk"}]}
    mock_resp = _mock_httpx_response(expected)
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.data.odds_client.httpx.AsyncClient", lambda: mock_http)
        result = await client.fetch_event_odds("ev1")
    assert result == expected


@pytest.mark.asyncio
async def test_fetch_scores_success():
    client = _make_client()
    expected = [{"id": "s1", "completed": True}]
    mock_resp = _mock_httpx_response(expected)
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.data.odds_client.httpx.AsyncClient", lambda: mock_http)
        result = await client.fetch_scores(days_from=2)
    assert result == expected


@pytest.mark.asyncio
async def test_fetch_player_props_success():
    client = _make_client()
    expected = {"bookmakers": [{"key": "fd"}]}
    mock_resp = _mock_httpx_response(expected)
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.data.odds_client.httpx.AsyncClient", lambda: mock_http)
        result = await client.fetch_player_props("ev2")
    assert result == expected


# ── persist_odds edge cases ────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_odds_no_id_skipped():
    """Events without 'id' are skipped and logged."""
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()
    # Event has no "id" key
    count = await client.persist_odds([{"bookmakers": []}], db)
    assert count == 0


@pytest.mark.asyncio
async def test_persist_odds_invalid_price():
    """Outcomes with non-numeric price are skipped."""
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = 42
    db.execute = AsyncMock(return_value=result_mock)

    data = [{
        "id": "ev1",
        "bookmakers": [{
            "key": "bk1",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "A", "price": "not_a_number"},
                    {"name": "B", "price": None},
                    {"name": "", "price": 100},  # empty name
                    {"name": "C", "price": 150, "point": "bad_point"},
                ],
            }],
        }],
    }]
    count = await client.persist_odds(data, db)
    # Only outcome "C" has valid price + name (point becomes None)
    assert count == 1


@pytest.mark.asyncio
async def test_persist_odds_high_skip_ratio():
    """When >50% events have no matching game, logger.error is used."""
    client = OddsClient()
    db = AsyncMock()
    db.add = MagicMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None  # no game match
    db.execute = AsyncMock(return_value=result_mock)

    data = [{"id": "ev1"}, {"id": "ev2"}, {"id": "ev3"}]
    count = await client.persist_odds(data, db)
    assert count == 0
