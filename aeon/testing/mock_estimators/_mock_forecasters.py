"""Mock forecasters useful for testing and debugging.

Used in tests for the forecasting base class.
"""

from aeon.forecasting.base import BaseForecaster


class MockForecaster(BaseForecaster):
    """Mock segmenter for testing."""

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None):
        return self

    def _predict(self, y):
        return 1.0

    def _forecast(self, y, X=None):
        return 1.0
