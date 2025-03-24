"""DummyForecaster always predicts the last value seen in training."""

from aeon.forecasting.base import BaseForecaster


class DummyForecaster(BaseForecaster):
    """Dummy forecaster always predicts the last value seen in training."""

    def __init__(self):
        """Initialize DummyForecaster."""
        self.last_value_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit dummy forecaster."""
        y = y.squeeze()
        self.last_value_ = y[-1]
        return self

    def _predict(self, y=None, exog=None):
        """Predict using dummy forecaster."""
        return self.last_value_

    def _forecast(self, y, exog=None):
        """Forecast using dummy forecaster."""
        y = y.squeeze()
        return y[-1]
