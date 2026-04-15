import pytest


def test_settings_defaults():
    from src.config import Settings

    s = Settings(odds_api_key="test", basketball_api_key="test")
    assert s.odds_api_sport_key == "basketball_nba"
    assert s.basketball_api_league_id == 12
    assert s.odds_fg_interval == 15
    assert s.odds_1h_interval == 30
    assert s.stats_interval == 360
    assert s.retrain_hour == 6
    assert s.odds_freshness_max_age_minutes == 365


def test_settings_ignores_shell_env(monkeypatch):
    # os.environ must NOT bleed into Settings — the profile file is the sole source.
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ODDS_API_KEY", "stale_shell_value")
    monkeypatch.setenv("BASKETBALL_API_KEY", "stale_shell_value")
    from src.config import Settings

    # Explicit init kwargs (representing profile-file values) win.
    s = Settings(app_env="test", odds_api_key="file_value", basketball_api_key="file_value")
    assert s.app_env == "test"
    assert s.odds_api_key == "file_value"
    assert s.basketball_api_key == "file_value"


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


def test_resolve_database_url_ignores_env_var(monkeypatch):
    """Stale DATABASE_URL in os.environ must NOT bleed into resolve_database_url."""
    from src.config import get_settings, resolve_database_url

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://stale:stale@stale:5432/stale")
    # Force .env loader to return no values so the fallback path is deterministic.
    monkeypatch.setattr("src.config.load_selected_env_values", lambda: {})
    get_settings.cache_clear()

    try:
        url = resolve_database_url()
        # Must come from Settings default, NOT from the stale env var.
        assert "stale" not in url
    finally:
        get_settings.cache_clear()
