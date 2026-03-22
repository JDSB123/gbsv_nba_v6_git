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


def test_validation_skipped_in_test_env():
    """APP_ENV=test should not require real API keys."""
    from src.config import Settings

    s = Settings(app_env="test", odds_api_key="", basketball_api_key="")
    assert s.odds_api_key == ""
    assert s.basketball_api_key == ""


def test_validation_fails_when_keys_missing():
    """Non-test envs must supply API keys."""
    from src.config import Settings

    with pytest.raises(ValueError, match="Missing required env vars"):
        Settings(app_env="development", odds_api_key="", basketball_api_key="")


def test_resolve_database_url_prefers_env_without_api_keys(monkeypatch):
    from src.config import get_settings, resolve_database_url

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db.example:5432/nba")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("BASKETBALL_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        assert (
            resolve_database_url()
            == "postgresql+asyncpg://user:pass@db.example:5432/nba"
        )
    finally:
        get_settings.cache_clear()
