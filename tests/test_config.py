import pytest


def test_settings_defaults():
    from src.config import Settings

    s = Settings(odds_api_key="test", basketball_api_key="test")
    assert s.odds_api_sport_key == "basketball_nba"
    assert s.basketball_api_league_id == 12
    assert s.odds_fg_interval == 15
    assert s.odds_1h_interval == 30
    assert s.stats_interval == 120
    assert s.retrain_hour == 6


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ODDS_API_KEY", "k1")
    monkeypatch.setenv("BASKETBALL_API_KEY", "k2")
    from src.config import Settings

    s = Settings()
    assert s.app_env == "production"
    assert s.odds_api_key == "k1"
    assert s.basketball_api_key == "k2"
