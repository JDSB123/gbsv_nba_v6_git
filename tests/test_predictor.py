import math

from src.models.features import get_feature_columns
from src.models.predictor import _margin_to_prob


def test_margin_to_prob_zero():
    assert _margin_to_prob(0) == 0.5


def test_margin_to_prob_positive():
    # Positive margin → > 50% win probability
    prob = _margin_to_prob(7.5)
    assert prob > 0.5
    assert prob < 1.0
    # At scale=7.5, margin=7.5 should be ~0.731
    assert abs(prob - (1 / (1 + math.exp(-1)))) < 0.01


def test_margin_to_prob_negative():
    prob = _margin_to_prob(-10)
    assert prob < 0.5


def test_feature_columns_count():
    cols = get_feature_columns()
    # 2 × (8 season + 8 recent + 5 schedule/injury) + 2 × 5 new per-team
    # + 12 game-level + 7 market + 12 sharp/square = 83
    assert len(cols) == 83


def test_feature_columns_no_duplicates():
    cols = get_feature_columns()
    assert len(cols) == len(set(cols))
