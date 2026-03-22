from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

NBA_SEASON_START_MONTH = 10
NBA_SEASON_START_DAY = 1
NBA_SEASON_END_MONTH = 6
NBA_SEASON_END_DAY = 30


def format_season(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def season_for_date(value: date | datetime) -> str:
    game_date = value.date() if isinstance(value, datetime) else value
    start_year = (
        game_date.year
        if game_date.month >= NBA_SEASON_START_MONTH
        else game_date.year - 1
    )
    return format_season(start_year)


def current_nba_season(today: date | None = None) -> str:
    return season_for_date(today or date.today())


def parse_season(season: str) -> tuple[int, int]:
    start_raw, end_raw = season.split("-", maxsplit=1)
    start_year = int(start_raw)
    end_year = int(end_raw)
    if end_year != start_year + 1:
        raise ValueError(f"Invalid NBA season '{season}'")
    return start_year, end_year


def season_bounds(season: str) -> tuple[date, date]:
    start_year, end_year = parse_season(season)
    season_start = date(start_year, NBA_SEASON_START_MONTH, NBA_SEASON_START_DAY)
    season_end = date(end_year, NBA_SEASON_END_MONTH, NBA_SEASON_END_DAY)
    return season_start, season_end


def resolve_backfill_window(
    season: str | None,
    days_back: int,
    today: date | None = None,
) -> tuple[str, date, date]:
    resolved_season = season or current_nba_season(today)
    season_start, season_end = season_bounds(resolved_season)
    anchor_date = today or date.today()
    end_date = min(max(anchor_date, season_start), season_end)
    start_date = max(season_start, end_date - timedelta(days=max(days_back, 0)))
    return resolved_season, start_date, end_date


def parse_api_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)
