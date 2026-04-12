"""Tests for src.notifications._text — plain-text slate builder."""

from datetime import UTC, datetime
from types import SimpleNamespace

from src.notifications._text import _format_game_line, build_teams_text


def _game(home="Lakers", away="Celtics", commence=None):
    return SimpleNamespace(
        home_team=SimpleNamespace(name=home),
        away_team=SimpleNamespace(name=away),
        home_team_id=1,
        away_team_id=2,
        commence_time=commence or datetime(2025, 3, 15, 23, 0, tzinfo=UTC),
    )


def _pred(fg_spread=-5.0, fg_total=222.0, fg_home_ml_prob=0.65):
    return SimpleNamespace(fg_spread=fg_spread, fg_total=fg_total, fg_home_ml_prob=fg_home_ml_prob)


class TestFormatGameLine:
    def test_basic_format(self):
        line = _format_game_line(_pred(), _game())
        assert "Celtics" in line
        assert "Lakers" in line
        assert "FG spread" in line
        assert "FG total" in line

    def test_contains_probabilities(self):
        line = _format_game_line(_pred(fg_home_ml_prob=0.70), _game())
        assert "0.700" in line
        assert "0.300" in line

    def test_none_teams_fallback(self):
        game = SimpleNamespace(
            home_team=None, away_team=None,
            home_team_id=10, away_team_id=20,
            commence_time=datetime(2025, 3, 15, 23, 0, tzinfo=UTC),
        )
        line = _format_game_line(_pred(), game)
        assert "Team 10" in line
        assert "Team 20" in line

    def test_none_commence_time(self):
        game = _game()
        game.commence_time = None
        line = _format_game_line(_pred(), game)
        assert "TBD" in line

    def test_null_spread_and_total(self):
        line = _format_game_line(_pred(fg_spread=None, fg_total=None, fg_home_ml_prob=None), _game())
        assert "FG spread +0.0" in line
        assert "FG total 0.0" in line


class TestBuildTeamsText:
    def test_header(self):
        result = build_teams_text([(_pred(), _game())], max_games=5)
        assert "NBA Predictions Update" in result

    def test_game_count(self):
        pairs = [(_pred(), _game()), (_pred(), _game("Hawks", "Nets"))]
        result = build_teams_text(pairs, max_games=5)
        assert "Games: 2" in result

    def test_max_games_limits_output(self):
        pairs = [(_pred(), _game()), (_pred(), _game("A", "B")), (_pred(), _game("C", "D"))]
        result = build_teams_text(pairs, max_games=1)
        assert "Games: 1" in result
        # Only first game should appear in body
        lines = result.split("\n")
        game_lines = [l for l in lines if l.startswith("- ")]
        assert len(game_lines) == 1

    def test_empty_input(self):
        result = build_teams_text([], max_games=5)
        assert "Games: 0" in result
