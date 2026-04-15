from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ACTIVE_ENV_PROFILE_FILE = ".env.profile"


def _is_placeholder_value(value: str) -> bool:
    return "placeholder" in value.lower()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_profile_file_selection() -> str:
    profile_path = _project_root() / ACTIVE_ENV_PROFILE_FILE
    if not profile_path.exists():
        return ""

    raw_selection = profile_path.read_text(encoding="utf-8").strip()
    if not raw_selection:
        return ""

    selection_path = Path(raw_selection)
    if selection_path.is_absolute():
        return raw_selection if selection_path.exists() else ""

    resolved = _project_root() / selection_path
    if resolved.exists():
        return raw_selection

    return ""


def resolve_settings_env_file() -> str:
    profile_selected_file = _resolve_profile_file_selection()
    if profile_selected_file:
        return profile_selected_file

    return ".env"


def load_selected_env_values() -> dict[str, str]:
    env_file = resolve_settings_env_file()
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = _project_root() / env_path
    if not env_path.exists():
        return {}

    values = dotenv_values(env_path)
    return {key: str(value) for key, value in values.items() if value is not None}


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
    stats_interval: int = 360
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

    # ── Odds API data source config ──────────────────────────────
    odds_api_regions: str = "us,us2,eu"  # us=retail, us2=offshore, eu=Pinnacle/bet365
    odds_api_markets_fg: str = "h2h,spreads,totals"
    odds_api_markets_1h: str = "h2h_h1,spreads_h1,totals_h1"
    odds_api_quota_min: int = 50  # skip fetches if remaining < this

    # ── Database pool ──────────────────────────────────────────────
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_ssl: bool = True  # set DB_SSL=false for local Postgres without SSL

    # ── Prediction reliability ────────────────────────────────────
    odds_freshness_max_age_minutes: int = 365

    # ── NBA constants ─────────────────────────────────────────────
    nba_avg_total: float = 230.0  # league-average total for edge calcs
    min_edge: float = 6.0  # minimum edge (pts) for a pick to qualify (legacy fallback)
    min_edge_spread: float = 4.5  # spread market threshold (pts)
    min_edge_total: float = 4.0  # total market threshold (pts)
    min_edge_ml: float = 5.0  # moneyline market threshold (EV pts)
    edge_thresholds: list[float] = [6.0, 7.0, 8.5, 10.0, 12.0]
    american_vig: int = 110  # standard -110 vig
    market_blend_alpha: float = 0.50  # spread blend: 50% model, 50% market
    server_port: int = 8000

    # ── Model governance / promotion gates ─────────────────────
    model_gate_min_rows: int = 200
    model_gate_max_mae_fg: float = 10.5
    model_gate_max_mae_1h: float = 7.5
    model_gate_max_rmse_fg: float = 14.0
    model_gate_max_rmse_1h: float = 9.5

    # ── CORS ──────────────────────────────────────────────────────
    cors_origins: list[str] = []

    # ── Azure (production) ────────────────────────────────────────
    azure_key_vault_url: str = ""
    applicationinsights_connection_string: str = ""
    azure_storage_connection_string: str = ""
    azure_storage_account_url: str = ""

    model_config = SettingsConfigDict(
        env_file=None,  # we own file loading via file_overrides in get_settings()
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        **kwargs,
    ):
        # Selected profile file (loaded as init kwargs) is the sole source of truth.
        # os.environ is excluded so stale shell vars cannot drift over the active profile.
        return (init_settings,)

    @model_validator(mode="after")
    def _check_required_secrets(self) -> Settings:
        """Fail fast if required secrets are missing outside test env."""
        if self.app_env == "test":
            return self

        missing: list[str] = []
        if not self.odds_api_key or _is_placeholder_value(self.odds_api_key):
            missing.append("ODDS_API_KEY")
        if not self.basketball_api_key or _is_placeholder_value(self.basketball_api_key):
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
    selected_values = load_selected_env_values()
    file_overrides: dict[str, Any] = {key.lower(): value for key, value in selected_values.items()}
    return Settings(**file_overrides)


def get_nba_avg_total() -> float:
    """Single source of truth for NBA average total."""
    try:
        return get_settings().nba_avg_total
    except Exception:
        return 230.0


def resolve_database_url() -> str:
    """Return the database URL without forcing unrelated secret validation."""
    selected_values = load_selected_env_values()
    selected_database_url = selected_values.get("DATABASE_URL", "").strip()
    if selected_database_url:
        return selected_database_url
    # Read the field default directly — avoids triggering the secrets validator.
    return Settings.model_fields["database_url"].default
