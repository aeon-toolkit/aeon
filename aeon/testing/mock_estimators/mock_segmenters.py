"""Mock segmenters for testing."""

from aeon.segmentation import BaseSegmenter


class MockSegmenter(BaseSegmenter):
    """Mock segmenter for testing."""

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """Generate breakpoints."""
        return [1]


class SupervisedMockSegmenter(MockSegmenter):
    """Mock segmenter for testing."""

    _tags = {
        "requires_y": True,
    }
