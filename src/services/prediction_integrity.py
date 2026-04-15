import math
from datetime import datetime
from typing import Any, cast

_PAYLOAD_TOLERANCE = 0.2
_MIN_FULL_GAME_SCORE = 60.0
_MAX_FULL_GAME_SCORE = 180.0
_MIN_FIRST_HALF_SCORE = 20.0
_MAX_FIRST_HALF_SCORE = 100.0
_MIN_FULL_GAME_TOTAL = 140.0
_MAX_FULL_GAME_TOTAL = 320.0
_MIN_FIRST_HALF_TOTAL = 60.0
_MAX_FIRST_HALF_TOTAL = 200.0
_NUMERIC_PREDICTION_FIELDS = (
    "predicted_home_fg",
    "predicted_away_fg",
    "predicted_home_1h",
    "predicted_away_1h",
    "fg_spread",
    "fg_total",
    "h1_spread",
    "h1_total",
)


def _parse_captured_at(prediction: Any) -> datetime | None:
    odds_sourced = cast(Any, getattr(prediction, "odds_sourced", None))
    captured_at = odds_sourced.get("captured_at") if isinstance(odds_sourced, dict) else None
    if not isinstance(captured_at, str) or not captured_at:
        return None
    try:
        return datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError:
        return None


def prediction_has_valid_score_payload(prediction: Any) -> bool:
    numeric_values: dict[str, float] = {}
    for field in _NUMERIC_PREDICTION_FIELDS:
        try:
            value = float(cast(Any, getattr(prediction, field, None)))
        except TypeError, ValueError:
            return False
        if not math.isfinite(value):
            return False
        numeric_values[field] = value

    if (
        numeric_values["predicted_home_fg"] < 0
        or numeric_values["predicted_away_fg"] < 0
        or numeric_values["predicted_home_1h"] < 0
        or numeric_values["predicted_away_1h"] < 0
    ):
        return False

    if not (
        _MIN_FULL_GAME_SCORE <= numeric_values["predicted_home_fg"] <= _MAX_FULL_GAME_SCORE
        and _MIN_FULL_GAME_SCORE <= numeric_values["predicted_away_fg"] <= _MAX_FULL_GAME_SCORE
        and _MIN_FIRST_HALF_SCORE <= numeric_values["predicted_home_1h"] <= _MAX_FIRST_HALF_SCORE
        and _MIN_FIRST_HALF_SCORE <= numeric_values["predicted_away_1h"] <= _MAX_FIRST_HALF_SCORE
        and _MIN_FULL_GAME_TOTAL <= numeric_values["fg_total"] <= _MAX_FULL_GAME_TOTAL
        and _MIN_FIRST_HALF_TOTAL <= numeric_values["h1_total"] <= _MAX_FIRST_HALF_TOTAL
    ):
        return False

    if (
        numeric_values["predicted_home_1h"] > numeric_values["predicted_home_fg"]
        or numeric_values["predicted_away_1h"] > numeric_values["predicted_away_fg"]
    ):
        return False

    return (
        abs(
            (numeric_values["predicted_home_fg"] - numeric_values["predicted_away_fg"])
            - numeric_values["fg_spread"]
        )
        <= _PAYLOAD_TOLERANCE
        and abs(
            (numeric_values["predicted_home_fg"] + numeric_values["predicted_away_fg"])
            - numeric_values["fg_total"]
        )
        <= _PAYLOAD_TOLERANCE
        and abs(
            (numeric_values["predicted_home_1h"] - numeric_values["predicted_away_1h"])
            - numeric_values["h1_spread"]
        )
        <= _PAYLOAD_TOLERANCE
        and abs(
            (numeric_values["predicted_home_1h"] + numeric_values["predicted_away_1h"])
            - numeric_values["h1_total"]
        )
        <= _PAYLOAD_TOLERANCE
    )


def prediction_has_valid_payload(prediction: Any) -> bool:
    return _parse_captured_at(prediction) is not None and prediction_has_valid_score_payload(
        prediction
    )


def prediction_payload_has_integrity_issues(prediction: Any) -> bool:
    return not prediction_has_valid_payload(prediction)


def _predicted_at_value(prediction: Any) -> datetime:
    predicted_at = cast(Any, getattr(prediction, "predicted_at", None))
    if not isinstance(predicted_at, datetime):
        return datetime.min
    if predicted_at.tzinfo is not None:
        return predicted_at.replace(tzinfo=None)
    return predicted_at


def prediction_score_rank(prediction: Any) -> tuple[int, datetime]:
    return (
        1 if prediction_has_valid_score_payload(prediction) else 0,
        _predicted_at_value(prediction),
    )


def prediction_rank(prediction: Any) -> tuple[int, datetime]:
    return (
        1 if prediction_has_valid_payload(prediction) else 0,
        _predicted_at_value(prediction),
    )
