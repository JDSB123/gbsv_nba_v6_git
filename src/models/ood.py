"""Out-of-distribution detection for NBA prediction inference.

Flags games whose feature vectors deviate significantly from the
training distribution, indicating the model is extrapolating and
predictions should be treated with lower confidence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Default threshold: Mahalanobis distance above which a sample is OOD.
# Roughly corresponds to 99th percentile of chi-squared distribution
# with ~100 degrees of freedom.
DEFAULT_OOD_THRESHOLD = 3.0


class OODDetector:
    """Detects out-of-distribution samples using Mahalanobis distance.

    Fits on the training feature matrix and flags inference samples
    that are far from the training centroid in covariance-adjusted space.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._threshold: float = DEFAULT_OOD_THRESHOLD
        self._training_distances: np.ndarray | None = None
        self._ready = False

    def fit(self, X: np.ndarray, threshold_percentile: float = 99.0) -> None:
        """Fit the detector on training data.

        Args:
            X: Training feature matrix (n_samples, n_features).
            threshold_percentile: Percentile of training distances
                above which samples are flagged as OOD.
        """
        if len(X) < 10:
            logger.warning("OOD detector: insufficient training data (%d rows)", len(X))
            return

        self._mean = np.mean(X, axis=0)

        # Regularised covariance to handle near-singular matrices
        cov = np.cov(X, rowvar=False)
        reg = np.eye(cov.shape[0]) * 1e-6
        try:
            self._inv_cov = np.linalg.inv(cov + reg)
        except np.linalg.LinAlgError:
            logger.warning("OOD detector: covariance matrix is singular, using diagonal")
            diag = np.diag(1.0 / (np.diag(cov) + 1e-6))
            self._inv_cov = diag

        # Compute training distances for threshold calibration
        distances = self._compute_distances(X)
        self._training_distances = distances
        self._threshold = float(np.percentile(distances, threshold_percentile))
        self._ready = True

        logger.info(
            "OOD detector fitted: mean_dist=%.2f, threshold(p%.0f)=%.2f",
            float(np.mean(distances)),
            threshold_percentile,
            self._threshold,
        )
        self._save()

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for each row."""
        if self._mean is None or self._inv_cov is None:
            return np.zeros(len(X))

        n_features = X.shape[1] if X.ndim == 2 else X.shape[0]
        if self._mean.shape[0] != n_features:
            logger.warning(
                "OOD detector dimension mismatch: mean has %d features, input has %d — skipping",
                self._mean.shape[0],
                n_features,
            )
            return np.zeros(len(X))

        diff = X - self._mean
        # Mahalanobis: sqrt( (x-μ)ᵀ Σ⁻¹ (x-μ) )
        left = diff @ self._inv_cov
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def score(self, X: np.ndarray) -> tuple[float, bool]:
        """Score a single sample for OOD-ness.

        Args:
            X: Feature vector (1, n_features).

        Returns:
            (distance, is_ood) tuple.
        """
        if not self._ready:
            return 0.0, False

        dist = float(self._compute_distances(X)[0])
        is_ood = dist > self._threshold
        return dist, is_ood

    def _save(self) -> None:
        if self._mean is None or self._inv_cov is None:
            return

        data = {
            "mean": self._mean.tolist(),
            "inv_cov": self._inv_cov.tolist(),
            "threshold": self._threshold,
        }
        path = ARTIFACTS_DIR / "ood_detector.json"
        path.write_text(json.dumps(data))
        logger.info("OOD detector saved to %s", path)

    def load(self) -> bool:
        path = ARTIFACTS_DIR / "ood_detector.json"
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            self._mean = np.array(data["mean"])
            self._inv_cov = np.array(data["inv_cov"])
            self._threshold = float(data["threshold"])
            self._ready = True
            logger.info("OOD detector loaded (threshold=%.2f)", self._threshold)
            return True
        except json.JSONDecodeError, KeyError, ValueError:
            logger.warning("Failed to load OOD detector", exc_info=True)
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready
