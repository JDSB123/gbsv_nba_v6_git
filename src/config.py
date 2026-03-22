from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_files() -> tuple[str, ...]:
    """Return env files to load, most-specific first.

    Pydantic loads files left-to-right; last file wins for dupes, but
    we list the *specific* file last so it overrides the base `.env`.
    Order: .env  →  .env.{APP_ENV}
    """
    env = os.getenv("APP_ENV", "development")
    specific = f".env.{env}"
    candidates: list[str] = [".env"]
    if Path(specific).is_file():
        candidates.append(specific)
    return tuple(candidates)


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

    # ── NBA constants ─────────────────────────────────────────────
    odds_api_sport_key: str = "basketball_nba"
    basketball_api_league_id: int = 12

    # ── Scheduler intervals (minutes) ────────────────────────────
    odds_fg_interval: int = 15
    odds_1h_interval: int = 30
    stats_interval: int = 120
    retrain_hour: int = 6  # 6am ET daily retrain

    # ── Quota management ──────────────────────────────────────────
    odds_api_quota_min: int = 50  # skip fetches if remaining < this

    # ── Azure (production) ────────────────────────────────────────
    azure_key_vault_url: str = ""

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _check_required_secrets(self) -> "Settings":
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
                f"Missing required env vars for env={self.app_env}: "
                + ", ".join(missing)
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
