"""Mock anomaly detectorsuseful for testing and debugging."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "MockAnomalyDetector",
    "MockAnomalyDetectorRequiresFit",
    "MockAnomalyDetectorRequiresY",
]


import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector


class MockAnomalyDetector(BaseAnomalyDetector):
    """Mock anomaly detector."""

    _tags = {
        "capability:missing_values": True,
        "capability:multivariate": True,
    }

    def __init__(self):
        super().__init__(axis=1)

    def _predict(self, X):
        return np.zeros(X.shape[self.axis])


class MockAnomalyDetectorRequiresFit(MockAnomalyDetector):
    """Mock anomaly detector that requires fit."""

    _tags = {
        "fit_is_empty": False,
    }

    def _fit(self, X, y=None):
        self._X = X
        return self


class MockAnomalyDetectorRequiresY(MockAnomalyDetectorRequiresFit):
    """Mock anomaly detector that requires y."""

    _tags = {
        "requires_y": True,
    }

    def _fit(self, X, y=None):
        self._X = X
        self._y = y
        return self
