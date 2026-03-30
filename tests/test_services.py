"""Tests for PredictionService and ModelService."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import Settings
from src.services.model import ModelService
from src.services.predictions import (
    PredictionService,
    _as_float,
    _parse_iso_utc,
)

# ══════════════════════════════════════════════════════════════════
#  _as_float / _parse_iso_utc
# ══════════════════════════════════════════════════════════════════


def test_as_float_with_number():
    assert _as_float(3.14) == 3.14


def test_as_float_none():
    assert _as_float(None) == 0.0


def test_as_float_none_custom_default():
    assert _as_float(None, 99.0) == 99.0


def test_parse_iso_utc_valid():
    ts = "2025-03-22T12:00:00Z"
    result = _parse_iso_utc(ts)
    assert result is not None
    assert result.tzinfo is not None
    assert result.year == 2025


def test_parse_iso_utc_with_offset():
    ts = "2025-03-22T12:00:00+05:00"
    result = _parse_iso_utc(ts)
    assert result is not None
    # Converted to UTC: 12:00+05:00 → 07:00 UTC
    assert result.hour == 7


def test_parse_iso_utc_invalid():
    assert _parse_iso_utc("not-a-date") is None


def test_parse_iso_utc_empty():
    assert _parse_iso_utc("") is None


def test_parse_iso_utc_none():
    assert _parse_iso_utc(None) is None


def test_parse_iso_utc_non_string():
    assert _parse_iso_utc(12345) is None


# ══════════════════════════════════════════════════════════════════
#  PredictionService.format_prediction
# ══════════════════════════════════════════════════════════════════


def _make_prediction(**overrides):
    captured_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    defaults = dict(
        game_id=1,
        predicted_home_fg=112.5,
        predicted_away_fg=108.3,
        predicted_home_1h=55.0,
        predicted_away_1h=53.0,
        fg_spread=4.2,
        fg_total=220.8,
        fg_home_ml_prob=0.62,
        h1_spread=2.0,
        h1_total=108.0,
        h1_home_ml_prob=0.58,
        model_version="v6.2.0",
        predicted_at=datetime(2025, 3, 22, 12, 0, tzinfo=UTC),
        odds_sourced={"captured_at": captured_at},
        opening_spread=-3.5,
        opening_total=219.0,
        closing_spread=-4.5,
        closing_total=221.0,
        clv_spread=0.3,
        clv_total=-0.2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_game(game_id=1, home_name="Boston Celtics", away_name="Los Angeles Lakers"):
    return SimpleNamespace(
        id=game_id,
        odds_api_id="ev123",
        commence_time=datetime(2025, 3, 22, 23, 0, tzinfo=UTC),
        home_team_id=1,
        away_team_id=2,
        home_team=SimpleNamespace(name=home_name),
        away_team=SimpleNamespace(name=away_name),
    )


def test_format_prediction_full():
    pred = _make_prediction()
    game = _make_game()
    svc = PredictionService(None, None, Settings())
    result = svc.format_prediction(pred, game)

    assert result["game_id"] == 1
    assert result["home_team"] == "Boston Celtics"
    assert result["away_team"] == "Los Angeles Lakers"
    assert result["predicted_scores"]["full_game"]["home"] == 112.5
    assert result["predicted_scores"]["first_half"]["away"] == 53.0
    assert result["markets"]["fg_spread"]["prediction"] == 4.2
    assert result["markets"]["fg_moneyline"]["home_prob"] == 0.62
    assert abs(result["markets"]["fg_moneyline"]["away_prob"] - 0.38) < 0.001
    assert result["model_version"] == "v6.2.0"
    assert result["clv"]["clv_spread"] == 0.3


def test_format_prediction_missing_team_names():
    pred = _make_prediction()
    game = SimpleNamespace(
        id=1,
        odds_api_id="ev1",
        commence_time=datetime(2025, 3, 22, tzinfo=UTC),
        home_team_id=1,
        away_team_id=2,
        home_team=None,
        away_team=None,
    )
    svc = PredictionService(None, None, Settings())
    result = svc.format_prediction(pred, game)
    assert result["home_team"] == "Team 1"
    assert result["away_team"] == "Team 2"


def test_format_prediction_none_commence_time():
    pred = _make_prediction()
    game = SimpleNamespace(
        id=1,
        odds_api_id="ev1",
        commence_time=None,
        home_team_id=1,
        away_team_id=2,
        home_team=SimpleNamespace(name="A"),
        away_team=SimpleNamespace(name="B"),
    )
    svc = PredictionService(None, None, Settings())
    result = svc.format_prediction(pred, game)
    assert result["commence_time"] is None


def test_format_prediction_none_predicted_at():
    pred = _make_prediction(predicted_at=None)
    game = _make_game()
    svc = PredictionService(None, None, Settings())
    result = svc.format_prediction(pred, game)
    assert result["predicted_at"] is None


def test_format_prediction_none_ml_probs_default_to_half():
    pred = _make_prediction(fg_home_ml_prob=None, h1_home_ml_prob=None)
    game = _make_game()
    svc = PredictionService(None, None, Settings())
    result = svc.format_prediction(pred, game)
    assert result["markets"]["fg_moneyline"]["home_prob"] == 0.5
    assert result["markets"]["h1_moneyline"]["home_prob"] == 0.5


# ══════════════════════════════════════════════════════════════════
#  PredictionService.evaluate_odds_freshness  (additional cases)
# ══════════════════════════════════════════════════════════════════


def test_freshness_all_ok():
    svc = PredictionService(None, None, Settings(odds_freshness_max_age_minutes=60))
    now_str = datetime.now(UTC).isoformat()
    rows = [
        {"odds_sourced": {"captured_at": now_str}},
        {"odds_sourced": {"captured_at": now_str}},
    ]
    result = svc.evaluate_odds_freshness(rows)
    assert result["status"] == "ok"
    assert result["stale_count"] == 0
    assert result["missing_odds_sourced"] == 0


def test_freshness_empty_list():
    svc = PredictionService(None, None, Settings(odds_freshness_max_age_minutes=30))
    result = svc.evaluate_odds_freshness([])
    assert result["status"] == "ok"
    assert result["evaluated_predictions"] == 0
    assert result["freshest_age_minutes"] is None


# ══════════════════════════════════════════════════════════════════
#  PredictionService.get_list_predictions  (mocked repo)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_list_predictions():
    pred = _make_prediction()
    game = _make_game()

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[pred])
    repo.get_games_with_teams = AsyncMock(return_value=[game])

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_list_predictions()

    assert result["count"] == 1
    assert result["predictions"][0]["game_id"] == 1
    assert "freshness" in result


@pytest.mark.asyncio
async def test_get_list_predictions_filters_stale_rows():
    stale_pred = _make_prediction(
        odds_sourced={
            "captured_at": (
                datetime.now(UTC) - timedelta(hours=2)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        },
    )
    game = _make_game()

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[stale_pred])
    repo.get_games_with_teams = AsyncMock(return_value=[game])

    svc = PredictionService(repo, None, Settings(odds_freshness_max_age_minutes=30))
    result = await svc.get_list_predictions()

    assert result["count"] == 0
    assert result["predictions"] == []
    assert result["freshness"]["filtered_out_non_fresh"] == 1


@pytest.mark.asyncio
async def test_get_list_predictions_empty():
    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[])
    repo.get_games_with_teams = AsyncMock(return_value=[])

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_list_predictions()

    assert result["count"] == 0
    assert result["predictions"] == []


@pytest.mark.asyncio
async def test_get_list_predictions_skips_predictions_without_loaded_games():
    pred1 = _make_prediction(game_id=1)
    pred2 = _make_prediction(game_id=2)
    game = _make_game(game_id=1)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[pred1, pred2])
    repo.get_games_with_teams = AsyncMock(return_value=[game])

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_list_predictions()

    assert result["count"] == 1
    assert result["predictions"][0]["game_id"] == 1


@pytest.mark.asyncio
async def test_get_list_predictions_skips_none_game_ids():
    pred1 = _make_prediction(game_id=None)
    pred2 = _make_prediction(game_id=2)
    game = _make_game(game_id=2)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[pred1, pred2])
    repo.get_games_with_teams = AsyncMock(return_value=[game])

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_list_predictions()

    assert result["count"] == 1
    assert result["predictions"][0]["game_id"] == 2


@pytest.mark.asyncio
async def test_get_list_predictions_skips_invalid_payloads():
    invalid_pred = _make_prediction(
        predicted_home_fg=-4.0,
        predicted_away_fg=100.0,
        predicted_home_1h=-2.0,
        predicted_away_1h=50.0,
        fg_spread=-104.0,
        fg_total=96.0,
        h1_spread=-52.0,
        h1_total=48.0,
    )
    game = _make_game()

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[invalid_pred])
    repo.get_games_with_teams = AsyncMock(return_value=[game])

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_list_predictions()

    assert result["count"] == 0
    assert result["predictions"] == []
    assert result["freshness"]["filtered_out_non_fresh"] == 0


# ══════════════════════════════════════════════════════════════════
#  PredictionService.get_slate_payload (mocked repo)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_slate_payload():
    pred = _make_prediction()
    game = _make_game()
    ts = datetime.now(UTC)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[pred])
    repo.get_games_with_teams_and_stats = AsyncMock(return_value=[game])
    repo.get_latest_odds_pull_timestamp = AsyncMock(return_value=ts)

    svc = PredictionService(repo, None, Settings())
    rows, odds_pulled = await svc.get_slate_payload()

    assert len(rows) == 1
    assert rows[0] == (pred, game)
    assert odds_pulled == ts


@pytest.mark.asyncio
async def test_get_slate_payload_filters_stale_predictions():
    stale_pred = _make_prediction(
        odds_sourced={
            "captured_at": (
                datetime.now(UTC) - timedelta(hours=2)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        },
    )
    game = _make_game()
    ts = datetime.now(UTC)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[stale_pred])
    repo.get_games_with_teams_and_stats = AsyncMock(return_value=[game])
    repo.get_latest_odds_pull_timestamp = AsyncMock(return_value=ts)

    svc = PredictionService(repo, None, Settings(odds_freshness_max_age_minutes=30))
    rows, odds_pulled = await svc.get_slate_payload()

    assert rows == []
    assert odds_pulled == ts


@pytest.mark.asyncio
async def test_get_slate_payload_skips_predictions_without_loaded_games():
    pred1 = _make_prediction(game_id=1)
    pred2 = _make_prediction(game_id=2)
    game = _make_game(game_id=1)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[pred1, pred2])
    repo.get_games_with_teams_and_stats = AsyncMock(return_value=[game])
    repo.get_latest_odds_pull_timestamp = AsyncMock(return_value=None)

    svc = PredictionService(repo, None, Settings())
    rows, odds_pulled = await svc.get_slate_payload()

    assert rows == [(pred1, game)]
    assert odds_pulled is None


@pytest.mark.asyncio
async def test_get_slate_payload_skips_none_game_ids():
    pred1 = _make_prediction(game_id=None)
    pred2 = _make_prediction(game_id=2)
    game = _make_game(game_id=2)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[pred1, pred2])
    repo.get_games_with_teams_and_stats = AsyncMock(return_value=[game])
    repo.get_latest_odds_pull_timestamp = AsyncMock(return_value=None)

    svc = PredictionService(repo, None, Settings())
    rows, odds_pulled = await svc.get_slate_payload()

    assert rows == [(pred2, game)]
    assert odds_pulled is None


@pytest.mark.asyncio
async def test_get_slate_payload_skips_invalid_predictions():
    invalid_pred = _make_prediction(
        predicted_home_fg=-4.0,
        predicted_away_fg=100.0,
        predicted_home_1h=-2.0,
        predicted_away_1h=50.0,
        fg_spread=-104.0,
        fg_total=96.0,
        h1_spread=-52.0,
        h1_total=48.0,
    )
    game = _make_game()
    ts = datetime.now(UTC)

    repo = AsyncMock()
    repo.get_latest_predictions_for_upcoming_games = AsyncMock(return_value=[invalid_pred])
    repo.get_games_with_teams_and_stats = AsyncMock(return_value=[game])
    repo.get_latest_odds_pull_timestamp = AsyncMock(return_value=ts)

    svc = PredictionService(repo, None, Settings())
    rows, odds_pulled = await svc.get_slate_payload()

    assert rows == []
    assert odds_pulled == ts


# ══════════════════════════════════════════════════════════════════
#  PredictionService.get_prediction_detail (mocked repo)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_prediction_detail_not_found():
    repo = AsyncMock()
    repo.get_game_with_teams = AsyncMock(return_value=None)

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_prediction_detail(999)
    assert result is None


@pytest.mark.asyncio
async def test_get_prediction_detail_no_prediction():
    game = _make_game()
    repo = AsyncMock()
    repo.get_game_with_teams = AsyncMock(return_value=game)
    repo.get_latest_prediction_for_game = AsyncMock(return_value=None)

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_prediction_detail(1)
    assert result["game"] == game
    assert result["pred"] is None


@pytest.mark.asyncio
async def test_get_prediction_detail_hides_stale_prediction():
    stale_pred = _make_prediction(
        odds_sourced={
            "captured_at": (
                datetime.now(UTC) - timedelta(hours=2)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        },
    )
    game = _make_game()
    repo = AsyncMock()
    repo.get_game_with_teams = AsyncMock(return_value=game)
    repo.get_latest_prediction_for_game = AsyncMock(return_value=stale_pred)

    svc = PredictionService(repo, None, Settings(odds_freshness_max_age_minutes=30))
    result = await svc.get_prediction_detail(1)

    assert result["game"] == game
    assert result["pred"] is None


@pytest.mark.asyncio
async def test_get_prediction_detail_with_odds():
    pred = _make_prediction()
    game = _make_game()
    snap_spread = SimpleNamespace(
        market="spreads",
        point=-4.5,
        outcome_name="Boston Celtics",
        bookmaker="fanduel",
        captured_at=datetime.now(UTC),
    )
    snap_total = SimpleNamespace(
        market="totals",
        point=220.0,
        outcome_name="Over",
        bookmaker="fanduel",
        captured_at=datetime.now(UTC),
    )

    repo = AsyncMock()
    repo.get_game_with_teams = AsyncMock(return_value=game)
    repo.get_latest_prediction_for_game = AsyncMock(return_value=pred)
    repo.get_recent_odds_snapshots = AsyncMock(return_value=[snap_spread, snap_total])

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_prediction_detail(1)
    assert "result" in result
    assert result["result"]["markets"]["fg_spread"]["market_line"] == -4.5
    assert result["result"]["markets"]["fg_total"]["market_line"] == 220.0


@pytest.mark.asyncio
async def test_get_prediction_detail_hides_invalid_prediction():
    invalid_pred = _make_prediction(
        predicted_home_fg=-4.0,
        predicted_away_fg=100.0,
        predicted_home_1h=-2.0,
        predicted_away_1h=50.0,
        fg_spread=-104.0,
        fg_total=96.0,
        h1_spread=-52.0,
        h1_total=48.0,
    )
    game = _make_game()
    repo = AsyncMock()
    repo.get_game_with_teams = AsyncMock(return_value=game)
    repo.get_latest_prediction_for_game = AsyncMock(return_value=invalid_pred)

    svc = PredictionService(repo, None, Settings())
    result = await svc.get_prediction_detail(1)

    assert result["game"] == game
    assert result["pred"] is None


# ══════════════════════════════════════════════════════════════════
#  ModelService.get_performance
# ══════════════════════════════════════════════════════════════════


def _make_perf_pred(
    game_id,
    model_version,
    home_fg=110,
    away_fg=105,
    home_1h=55,
    away_1h=52,
    fg_spread=None,
    fg_total=None,
    h1_spread=None,
    h1_total=None,
    clv_spread=0.5,
    clv_total=-0.3,
):
    if fg_spread is None:
        fg_spread = home_fg - away_fg
    if fg_total is None:
        fg_total = home_fg + away_fg
    if h1_spread is None:
        h1_spread = home_1h - away_1h
    if h1_total is None:
        h1_total = home_1h + away_1h
    return SimpleNamespace(
        game_id=game_id,
        model_version=model_version,
        predicted_home_fg=home_fg,
        predicted_away_fg=away_fg,
        predicted_home_1h=home_1h,
        predicted_away_1h=away_1h,
        fg_spread=fg_spread,
        fg_total=fg_total,
        h1_spread=h1_spread,
        h1_total=h1_total,
        clv_spread=clv_spread,
        clv_total=clv_total,
    )


def _make_perf_game(game_id, home_fg=112, away_fg=108, home_1h=56, away_1h=54):
    return SimpleNamespace(
        id=game_id,
        home_score_fg=home_fg,
        away_score_fg=away_fg,
        home_score_1h=home_1h,
        away_score_1h=away_1h,
    )


@pytest.mark.asyncio
async def test_model_service_get_performance():
    pred = _make_perf_pred(1, "v6.2.0")
    game = _make_perf_game(1)

    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=[(pred, game)])

    svc = ModelService(repo)
    result = await svc.get_performance(limit=200)

    assert result["window"] == 200
    assert len(result["models"]) == 1
    m = result["models"][0]
    assert m["model_version"] == "v6.2.0"
    assert m["sample_size"] == 1
    assert m["mae_home_fg"] is not None
    assert m["mae_away_fg"] is not None
    assert m["mae_home_1h"] is not None
    assert m["mae_away_1h"] is not None


@pytest.mark.asyncio
async def test_model_service_get_performance_empty():
    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=[])

    svc = ModelService(repo)
    result = await svc.get_performance()

    assert result["window"] == 200
    assert result["models"] == []


@pytest.mark.asyncio
async def test_model_service_respects_limit_while_collecting_rows():
    rows = [
        (_make_perf_pred(1, "v6.1.0"), _make_perf_game(1)),
        (_make_perf_pred(2, "v6.2.0"), _make_perf_game(2)),
    ]

    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=rows)

    svc = ModelService(repo)
    result = await svc.get_performance(limit=1)

    assert len(result["models"]) == 1
    assert result["models"][0]["model_version"] == "v6.1.0"


@pytest.mark.asyncio
async def test_model_service_skips_games_without_scores():
    pred = _make_perf_pred(1, "v6.2.0")
    game = SimpleNamespace(
        id=1, home_score_fg=None, away_score_fg=None, home_score_1h=None, away_score_1h=None
    )

    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=[(pred, game)])

    svc = ModelService(repo)
    result = await svc.get_performance()

    assert result["models"] == []


@pytest.mark.asyncio
async def test_model_service_multiple_versions():
    rows = [
        (_make_perf_pred(1, "v6.1.0"), _make_perf_game(1)),
        (_make_perf_pred(2, "v6.2.0"), _make_perf_game(2)),
        (_make_perf_pred(3, "v6.2.0"), _make_perf_game(3)),
    ]

    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=rows)

    svc = ModelService(repo)
    result = await svc.get_performance()

    versions = {m["model_version"] for m in result["models"]}
    assert versions == {"v6.1.0", "v6.2.0"}
    # v6.2.0 has 2 samples, should be first (sorted desc)
    assert result["models"][0]["sample_size"] == 2


@pytest.mark.asyncio
async def test_model_service_skips_implausible_score_payloads():
    invalid_pred = _make_perf_pred(
        1,
        "v6.2.0",
        home_fg=4.3,
        away_fg=7.8,
        home_1h=2.1,
        away_1h=5.3,
    )
    game = _make_perf_game(1)

    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=[(invalid_pred, game)])

    svc = ModelService(repo)
    result = await svc.get_performance()

    assert result["models"] == []


@pytest.mark.asyncio
async def test_model_service_handles_none_1h_scores():
    pred = _make_perf_pred(1, "v6.2.0")
    game = SimpleNamespace(
        id=1, home_score_fg=110, away_score_fg=105, home_score_1h=None, away_score_1h=None
    )

    repo = AsyncMock()
    repo.get_finished_game_predictions = AsyncMock(return_value=[(pred, game)])

    svc = ModelService(repo)
    result = await svc.get_performance()

    m = result["models"][0]
    assert m["mae_home_fg"] is not None
    assert m["mae_home_1h"] is None  # no 1H scores available
