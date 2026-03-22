from src.db.models import Base, Team, Game, OddsSnapshot, Prediction


def test_all_tables_defined():
    table_names = set(Base.metadata.tables.keys())
    expected = {
        "teams",
        "players",
        "games",
        "team_season_stats",
        "player_game_stats",
        "odds_snapshots",
        "predictions",
        "injuries",
    }
    assert expected.issubset(table_names)


def test_game_relationships():
    g = Game()
    assert hasattr(g, "home_team")
    assert hasattr(g, "away_team")
    assert hasattr(g, "odds_snapshots")
    assert hasattr(g, "predictions")
    assert hasattr(g, "player_stats")
