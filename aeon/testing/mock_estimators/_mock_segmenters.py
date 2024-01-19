"""Mock segmenters for testing."""
import numpy as np

from aeon.segmentation import BaseSegmenter


class MockSegmenter(BaseSegmenter):
    """Mock segmenter for testing."""

    _tags = {
        "capability:missing_values": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
    }

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """Generate breakpoints."""
        return np.array([1])


class SupervisedMockSegmenter(MockSegmenter):
    """Mock segmenter for testing."""

    _tags = {
        "requires_y": True,
    }
