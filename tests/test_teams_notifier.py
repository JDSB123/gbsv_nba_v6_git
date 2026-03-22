from datetime import UTC, datetime
from types import SimpleNamespace

from src.notifications.teams import build_teams_text


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
