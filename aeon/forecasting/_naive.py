"""Naive Forecaster."""

from aeon.forecasting.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Naive forecaster that always predicts the last value seen in training."""

    def __init__(self):
        """Initialize NaiveForecaster."""
        self.last_value_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit Naive forecaster."""
        y = y.squeeze()
        self.last_value_ = y[-1]
        return self

    def _predict(self, exog=None):
        """Predict using Naive forecaster."""
        return self.last_value_
