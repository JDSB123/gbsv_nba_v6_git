"""Tests for trainer – pure-logic functions that don't require a DB."""

import numpy as np

from src.models.trainer import (
    DEFAULT_XGB_PARAMS,
    MODEL_NAMES,
    TARGETS,
    _calibrate_probabilities,
    _evaluate_promotion,
)

# ── _evaluate_promotion ───────────────────────────────────────

def test_evaluate_promotion_passes():
    metrics = {
        "model_home_fg_mae": 8.0,
        "model_away_fg_mae": 8.0,
        "model_home_1h_mae": 6.0,
        "model_away_1h_mae": 6.0,
        "model_home_fg_rmse": 10.0,
        "model_away_fg_rmse": 10.0,
        "model_home_1h_rmse": 7.0,
        "model_away_1h_rmse": 7.0,
    }
    ok, reason = _evaluate_promotion(metrics, row_count=500)
    assert ok is True
    assert "all gate checks passed" in reason


def test_evaluate_promotion_insufficient_rows():
    metrics = {}
    ok, reason = _evaluate_promotion(metrics, row_count=10)
    assert ok is False
    assert "insufficient rows" in reason


def test_evaluate_promotion_high_mae():
    metrics = {
        "model_home_fg_mae": 20.0,  # very high
        "model_away_fg_mae": 8.0,
        "model_home_1h_mae": 6.0,
        "model_away_1h_mae": 6.0,
        "model_home_fg_rmse": 10.0,
        "model_away_fg_rmse": 10.0,
        "model_home_1h_rmse": 7.0,
        "model_away_1h_rmse": 7.0,
    }
    ok, reason = _evaluate_promotion(metrics, row_count=500)
    assert ok is False
    assert "model_home_fg_mae" in reason


def test_evaluate_promotion_missing_metric():
    metrics = {
        "model_home_fg_mae": 8.0,
        # everything else missing
    }
    ok, reason = _evaluate_promotion(metrics, row_count=500)
    assert ok is False
    assert "missing" in reason


# ── _calibrate_probabilities ──────────────────────────────────

def test_calibrate_probabilities_returns_floats():
    np.random.seed(42)
    margins = np.random.randn(100) * 10
    actuals = (margins > 0).astype(int)
    coef, intercept = _calibrate_probabilities(margins, actuals)
    assert isinstance(coef, float)
    assert isinstance(intercept, float)


def test_calibrate_probabilities_positive_coef():
    """When positive margin → win, coef should be positive."""
    margins = np.array([-15, -10, -5, 5, 10, 15], dtype=float)
    actuals = np.array([0, 0, 0, 1, 1, 1])
    coef, _ = _calibrate_probabilities(margins, actuals)
    assert coef > 0


# ── Constants ──────────────────────────────────────────────────

def test_targets_and_model_names_aligned():
    assert len(TARGETS) == len(MODEL_NAMES)
    assert len(TARGETS) == 4


def test_default_xgb_params():
    assert DEFAULT_XGB_PARAMS["objective"] == "reg:squarederror"
    assert DEFAULT_XGB_PARAMS["random_state"] == 42
    assert "max_depth" in DEFAULT_XGB_PARAMS
    assert "learning_rate" in DEFAULT_XGB_PARAMS
