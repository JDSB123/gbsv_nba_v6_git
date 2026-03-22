from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API keys
    odds_api_key: str = ""
    basketball_api_key: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nba_gbsv"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # API base URLs
    odds_api_base: str = "https://api.the-odds-api.com/v4"
    basketball_api_base: str = "https://v1.basketball.api-sports.io"

    # NBA constants
    odds_api_sport_key: str = "basketball_nba"
    basketball_api_league_id: int = 12

    # Scheduler intervals (minutes)
    odds_fg_interval: int = 15
    odds_1h_interval: int = 30
    stats_interval: int = 120
    retrain_hour: int = 6  # 6am ET daily retrain

    # Quota management
    odds_api_quota_min: int = 50  # skip fetches if remaining < this

    # Azure (production)
    azure_key_vault_url: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
