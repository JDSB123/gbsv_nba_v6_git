"""Tests for Predictor internals – status, calibration, compatibility mode."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.models.predictor import (
    ARTIFACTS_DIR,
    MODEL_NAMES,
    Predictor,
    _margin_to_prob,
)
from src.models.versioning import MODEL_VERSION


# ── _margin_to_prob (extended) ─────────────────────────────────

def test_margin_to_prob_with_calibration():
    # Platt-scaled: z = coef * margin + intercept
    prob = _margin_to_prob(5.0, coef=0.2, intercept=-0.5)
    expected = 1.0 / (1.0 + math.exp(-(0.2 * 5.0 - 0.5)))
    assert abs(prob - expected) < 1e-9


def test_margin_to_prob_calibration_zero_intercept():
    prob = _margin_to_prob(0.0, coef=0.2, intercept=0.0)
    assert prob == 0.5


def test_margin_to_prob_large_positive():
    prob = _margin_to_prob(50.0)
    assert prob > 0.99


def test_margin_to_prob_large_negative():
    prob = _margin_to_prob(-50.0)
    assert prob < 0.01


def test_margin_to_prob_symmetry():
    p_pos = _margin_to_prob(5.0)
    p_neg = _margin_to_prob(-5.0)
    assert abs(p_pos + p_neg - 1.0) < 1e-9


# ── Predictor construction ─────────────────────────────────────

def test_predictor_has_feature_cols():
    predictor = Predictor()
    assert len(predictor.feature_cols) == 122


def test_predictor_model_version():
    predictor = Predictor()
    assert predictor.model_version == MODEL_VERSION


def test_predictor_runtime_status_structure():
    predictor = Predictor()
    status = predictor.get_runtime_status()
    assert "ready" in status
    assert "expected_features" in status
    assert "loaded_models" in status
    assert "missing_models" in status
    assert "compatibility_mode" in status
    assert "inference_features" in status


def test_predictor_is_ready_when_artifacts_exist():
    """If artifacts exist on disk (which they do after training),
    predictor should be ready."""
    predictor = Predictor()
    # If artifacts directory has all 4 model jsons, should be ready
    all_exist = all((ARTIFACTS_DIR / f"{n}.json").exists() for n in MODEL_NAMES)
    assert predictor.is_ready == all_exist


# ── Calibration loading ────────────────────────────────────────

def test_predictor_loads_calibration():
    """If metrics.json has calibration keys, they should be loaded."""
    predictor = Predictor()
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        if "calibration_fg_coef" in m:
            assert "calibration_fg_coef" in predictor._calibration
    # No assertion needed if file doesn't exist — just verifying no crash
