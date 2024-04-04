"""Mock anomaly detectors for testing."""

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector


class MockAnomalyDetector(BaseAnomalyDetector):
    """Mock anomaly detector."""

    def __init__(self):
        super().__init__()

    _tags = {
        "capability:missing_values": True,
        "capability:multivariate": True,
    }

    def _predict(self, X):
        return np.zeros(len(X))
