"""SHAP-based model explainability for NBA predictions.

Provides per-prediction feature attribution (waterfall data) and
global feature importance ranking based on SHAP values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import shap

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Limit SHAP background dataset size for performance
_MAX_BACKGROUND_SAMPLES = 200


class Explainer:
    """Wraps SHAP TreeExplainer for XGBoost models."""

    def __init__(self) -> None:
        self._explainers: dict[str, shap.TreeExplainer] = {}
        self._feature_cols: list[str] = []
        self._global_importance: dict[str, dict[str, float]] = {}

    def initialize(
        self,
        models: dict[str, Any],
        feature_cols: list[str],
        background_X: np.ndarray | None = None,
    ) -> None:
        """Create TreeExplainers for each model.

        Args:
            models: XGBoost models keyed by model name.
            feature_cols: Ordered feature column names.
            background_X: Background dataset for SHAP (optional).
                When None, uses model internals (faster but less accurate
                for interaction effects).
        """
        self._feature_cols = list(feature_cols)
        for name, model in models.items():
            try:
                if background_X is not None:
                    bg = background_X
                    if len(bg) > _MAX_BACKGROUND_SAMPLES:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(len(bg), _MAX_BACKGROUND_SAMPLES, replace=False)
                        bg = bg[idx]
                    self._explainers[name] = shap.TreeExplainer(model, bg)
                else:
                    self._explainers[name] = shap.TreeExplainer(model)
                logger.info("SHAP explainer initialized for %s", name)
            except Exception:
                logger.warning("Failed to initialize SHAP explainer for %s", name, exc_info=True)

    def explain_prediction(
        self,
        X: np.ndarray,
        model_name: str,
    ) -> dict[str, Any] | None:
        """Compute SHAP values for a single prediction.

        Returns a dict with:
          - base_value: Expected model output
          - shap_values: {feature_name: shap_value} sorted by |importance|
          - top_drivers: Top 10 features driving this prediction
        """
        explainer = self._explainers.get(model_name)
        if explainer is None:
            return None

        try:
            sv = explainer.shap_values(X)
        except Exception:
            logger.warning("SHAP computation failed for %s", model_name, exc_info=True)
            return None

        if isinstance(sv, list):
            sv = sv[0]
        values = sv.flatten()

        if len(values) != len(self._feature_cols):
            logger.warning(
                "SHAP value count mismatch: %d vs %d features",
                len(values),
                len(self._feature_cols),
            )
            return None

        raw_base_value = explainer.expected_value
        if isinstance(raw_base_value, list):
            raw_base_value = raw_base_value[0]
        if isinstance(raw_base_value, np.ndarray):
            raw_base_value = raw_base_value.flatten()[0]
        base_value = float(raw_base_value)

        feature_shap = {
            name: round(float(val), 4) for name, val in zip(self._feature_cols, values, strict=True)
        }
        sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        top_drivers = [
            {"feature": name, "impact": val}
            for name, val in sorted_features[:10]
        ]

        return {
            "base_value": round(base_value, 2),
            "shap_values": dict(sorted_features),
            "top_drivers": top_drivers,
        }

    def compute_global_importance(
        self,
        X: np.ndarray,
        model_name: str,
    ) -> dict[str, float]:
        """Compute mean |SHAP| importance across a dataset.

        Saved to artifacts for feature pruning decisions.
        """
        explainer = self._explainers.get(model_name)
        if explainer is None:
            return {}

        # Limit dataset size for performance
        if len(X) > _MAX_BACKGROUND_SAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), _MAX_BACKGROUND_SAMPLES, replace=False)
            X = X[idx]

        try:
            sv = explainer.shap_values(X)
        except Exception:
            logger.warning("Global SHAP failed for %s", model_name, exc_info=True)
            return {}

        if isinstance(sv, list):
            sv = sv[0]

        mean_abs = np.mean(np.abs(sv), axis=0)
        importance = {
            name: round(float(val), 6)
            for name, val in zip(self._feature_cols, mean_abs, strict=True)
        }
        self._global_importance[model_name] = importance
        return importance

    def save_global_importance(self) -> None:
        if self._global_importance:
            path = ARTIFACTS_DIR / "shap_importance.json"
            path.write_text(json.dumps(self._global_importance, indent=2))
            logger.info("SHAP global importance saved to %s", path)

    def get_pruning_candidates(
        self,
        threshold_pct: float = 0.01,
    ) -> list[str]:
        """Return features whose mean |SHAP| is below threshold_pct of the max.

        These are candidates for removal to simplify the model.
        """
        if not self._global_importance:
            return []

        # Aggregate across all models
        aggregated: dict[str, float] = {}
        for imp in self._global_importance.values():
            for feat, val in imp.items():
                aggregated[feat] = aggregated.get(feat, 0.0) + val

        if not aggregated:
            return []

        max_imp = max(aggregated.values())
        if max_imp <= 0:
            return []

        threshold = max_imp * threshold_pct
        return sorted(
            [feat for feat, val in aggregated.items() if val < threshold],
            key=lambda f: aggregated[f],
        )

    @property
    def is_ready(self) -> bool:
        return bool(self._explainers)
