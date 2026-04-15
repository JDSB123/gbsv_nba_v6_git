from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_placeholder_value(value: str) -> bool:
    return "placeholder" in value.lower()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_settings_env_file() -> str:
    return ".env"


def load_selected_env_values() -> dict[str, str]:
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return {}

    values = dotenv_values(env_path)
    return {key: str(value) for key, value in values.items() if value is not None}


class Settings(BaseSettings):
    # ── API keys ──────────────────────────────────────────────────
    odds_api_key: str = Field(
        default="",
        json_schema_extra={
            "env_group": "Required secrets",
            "env_template": "dev-odds-placeholder",
            "env_comment": "Replace placeholders with real keys for live API calls.",
        },
    )
    basketball_api_key: str = Field(
        default="",
        json_schema_extra={
            "env_group": "Required secrets",
            "env_template": "dev-basketball-placeholder",
        },
    )

    # ── Database ──────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/nba_gbsv",
        json_schema_extra={
            "env_group": "Database",
            "env_comment": (
                "Dev container overrides this to postgresql+asyncpg://postgres:postgres@db:5432/nba_gbsv. "
                "Host-only local runs should keep localhost."
            ),
        },
    )

    # ── App ───────────────────────────────────────────────────────
    app_env: str = Field(
        default="development",
        json_schema_extra={"env_group": "App"},
    )
    log_level: str = Field(
        default="INFO",
        json_schema_extra={"env_group": "App"},
    )

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
    teams_webhook_url: str = Field(
        default="",
        json_schema_extra={
            "env_group": "Teams delivery",
            "env_comment": (
                "Preferred: Graph API posting via TEAMS_TEAM_ID + TEAMS_CHANNEL_ID. "
                "Fallback: legacy incoming webhook URL set here."
            ),
        },
    )
    teams_team_id: str = ""
    teams_channel_id: str = ""
    teams_max_games_per_message: int = 8

    # ── Public API base URL (used for download links in cards) ──
    api_base_url: str = Field(
        default="",
        json_schema_extra={"env_group": "App"},
    )

    # ── API authentication ────────────────────────────────────────
    api_key: str = Field(
        default="",
        json_schema_extra={
            "env_group": "App",
            "env_comment": "X-API-Key header; empty = no auth enforced.",
        },
    )

    # ── OneDrive export helper ────────────────────────────────────
    onedrive_export_root: str = Field(
        default="",
        json_schema_extra={
            "env_group": "App",
            "env_comment": "Used by scripts/export_onedrive.py only.",
        },
    )
    export_allow_local_db: bool = Field(
        default=False,
        json_schema_extra={
            "env_group": "App",
            "env_template": "false",
            "env_comment": "Set true to allow scripts/export_onedrive.py against a localhost DB.",
        },
    )

    # ── Odds API data source config ──────────────────────────────
    odds_api_regions: str = "us,us2,eu"  # us=retail, us2=offshore, eu=Pinnacle/bet365
    odds_api_markets_fg: str = "h2h,spreads,totals"
    odds_api_markets_1h: str = "h2h_h1,spreads_h1,totals_h1"
    odds_api_quota_min: int = 50  # skip fetches if remaining < this

    # ── Database pool ──────────────────────────────────────────────
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_ssl: bool = Field(
        default=True,
        json_schema_extra={
            "env_group": "Database",
            "env_template": "false",
            "env_comment": "Set DB_SSL=false for local Postgres; leave true for Azure Postgres.",
        },
    )

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
    azure_key_vault_url: str = Field(
        default="",
        json_schema_extra={"env_group": "Azure"},
    )
    applicationinsights_connection_string: str = Field(
        default="",
        json_schema_extra={"env_group": "Azure"},
    )
    azure_storage_connection_string: str = Field(
        default="",
        json_schema_extra={"env_group": "Azure"},
    )
    azure_storage_account_url: str = Field(
        default="",
        json_schema_extra={"env_group": "Azure"},
    )

    model_config = SettingsConfigDict(
        # .env is the single runtime env file. Actual loading goes through
        # get_settings() -> load_selected_env_values() so we control precedence
        # (init_settings only — os.environ is excluded below).
        env_file=".env",
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
        # When app_env was not explicitly passed, fall back to the process-level
        # APP_ENV (set by tests/conftest.py). settings_customise_sources excludes
        # os.environ from field loading, but the test-mode escape hatch should
        # still respect the running process's declared environment.
        effective_app_env = self.app_env
        if "app_env" not in self.model_fields_set:
            effective_app_env = os.getenv("APP_ENV", self.app_env)
        if effective_app_env == "test":
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


_ENV_GROUP_ORDER = (
    "Required secrets",
    "Database",
    "App",
    "Azure",
    "Teams delivery",
)


def env_template_entries() -> list[tuple[str, str, str, str | None]]:
    """Return the curated env-template entries declared on Settings.

    Each entry is (group, ENV_KEY, value, comment_or_None). Walks
    ``Settings.model_fields`` and picks up any field whose
    ``json_schema_extra`` carries an ``env_group`` marker. Field declaration
    order is preserved within each group; groups themselves are emitted in
    ``_ENV_GROUP_ORDER`` order.
    """
    buckets: dict[str, list[tuple[str, str, str, str | None]]] = {
        group: [] for group in _ENV_GROUP_ORDER
    }
    for field_name, field in Settings.model_fields.items():
        extra = field.json_schema_extra or {}
        if not isinstance(extra, dict):
            continue
        group = extra.get("env_group")
        if not group:
            continue
        env_key = field_name.upper()
        if "env_template" in extra:
            value = str(extra["env_template"])
        elif field.default is None:
            value = ""
        else:
            value = str(field.default)
        comment = extra.get("env_comment")
        buckets.setdefault(group, []).append((group, env_key, value, comment))

    ordered: list[tuple[str, str, str, str | None]] = []
    for group in _ENV_GROUP_ORDER:
        ordered.extend(buckets.get(group, []))
    # Any custom group names declared on fields but missing from the order list
    # still get rendered, in alphabetical order, so nothing is silently dropped.
    for group in sorted(buckets):
        if group in _ENV_GROUP_ORDER:
            continue
        ordered.extend(buckets[group])
    return ordered


def generate_env_template() -> str:
    """Render the .env template from Settings field metadata.

    src/config.py is the single source of truth for the env schema. This
    function is what scripts/sync_env.py uses to materialize .env — there is
    no separate .env.example file in the repo.
    """
    lines: list[str] = [
        "# ----------------------------------------------------------------",
        "# NBA GBSV v6 - .env",
        "# Generated by scripts/sync_env.py from src/config.py (Settings).",
        "# Edit values here to override defaults locally.",
        "# Re-run scripts/sync_env.py to repair missing keys after pulls.",
        "# ----------------------------------------------------------------",
        "",
    ]
    current_group: str | None = None
    for group, key, value, comment in env_template_entries():
        if group != current_group:
            if current_group is not None:
                lines.append("")
            lines.append(f"# {group}")
            current_group = group
        if comment:
            lines.append(f"# {comment}")
        lines.append(f"{key}={value}")
    return "\n".join(lines).rstrip() + "\n"
