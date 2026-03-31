from __future__ import annotations

import os
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── API keys ──────────────────────────────────────────────────
    odds_api_key: str = ""
    basketball_api_key: str = ""

    # ── Database ──────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nba_gbsv"

    # ── App ───────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── API base URLs ─────────────────────────────────────────────
    odds_api_base: str = "https://api.the-odds-api.com/v4"
    basketball_api_base: str = "https://v1.basketball.api-sports.io"
    nba_api_base: str = "https://v2.nba.api-sports.io"

    # ── NBA constants ─────────────────────────────────────────────
    odds_api_sport_key: str = "basketball_nba"
    basketball_api_league_id: int = 12

    # ── Scheduler intervals (minutes) ────────────────────────────
    odds_fg_interval: int = 15
    odds_1h_interval: int = 30
    stats_interval: int = 120
    injuries_interval: int = 120
    retrain_hour: int = 6  # 6am ET daily retrain
    morning_slate_hour: int = 10  # 10am ET = 9am CST daily publish
    pregame_lead_minutes: int = 60  # publish this many min before first tip
    prediction_cache_refresh_interval_minutes: int = 15

    # ── Teams delivery ─────────────────────────────────────────
    teams_webhook_url: str = ""
    teams_team_id: str = ""
    teams_channel_id: str = ""
    teams_max_games_per_message: int = 8

    # ── Public API base URL (used for download links in cards) ──
    api_base_url: str = ""

    # ── API authentication ────────────────────────────────────────
    api_key: str = ""  # X-API-Key header; empty = no auth enforced

    # ── Quota management ──────────────────────────────────────────
    odds_api_quota_min: int = 50  # skip fetches if remaining < this

    # ── Database pool ──────────────────────────────────────────────
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # ── Prediction reliability ────────────────────────────────────
    odds_freshness_max_age_minutes: int = 30

    # ── NBA constants (notifications) ─────────────────────────────
    nba_avg_total: float = 230.0  # league-average total for edge calcs

    # ── Model governance / promotion gates ─────────────────────
    model_gate_min_rows: int = 200
    model_gate_max_mae_fg: float = 13.0
    model_gate_max_mae_1h: float = 8.0
    model_gate_max_rmse_fg: float = 16.0
    model_gate_max_rmse_1h: float = 10.0

    # ── Azure (production) ────────────────────────────────────────
    # Placeholder for Azure Key Vault integration — not yet wired up.
    # When set, secrets (API keys, DB URL) will be fetched from Key Vault at startup.
    azure_key_vault_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _check_required_secrets(self) -> Settings:
        """Fail fast if required secrets are missing outside test env."""
        if self.app_env == "test":
            return self
        missing: list[str] = []
        if not self.odds_api_key:
            missing.append("ODDS_API_KEY")
        if not self.basketball_api_key:
            missing.append("BASKETBALL_API_KEY")
        if not self.database_url:
            missing.append("DATABASE_URL")
        if missing:
            raise ValueError(
                f"Missing required env vars for env={self.app_env}: " + ", ".join(missing)
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_database_url() -> str:
    """Return the database URL without forcing unrelated secret validation."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url
    return get_settings().database_url
