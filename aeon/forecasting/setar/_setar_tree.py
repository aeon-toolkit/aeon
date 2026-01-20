"""SETAR Tree forecaster."""

from aeon.forecasting.setar._setar import SETARForecaster


class SETARTreeForecaster:
    """Apply SETARForecaster independently to multiple series."""

    def __init__(self, lags: int = 1, threshold_lag: int = 1):
        self.lags = lags
        self.threshold_lag = threshold_lag

    def fit(self, y):
        """Fit one SETAR model per series."""
        self.models_ = []

        for series in y:
            model = SETARForecaster(lags=self.lags, threshold_lag=self.threshold_lag)
            model.fit(series)
            self.models_.append(model)

        return self

    def predict(self, fh):
        """Predict using the first fitted model."""
        return self.models_[0].predict(fh)
