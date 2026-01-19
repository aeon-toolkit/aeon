"""Global SETAR-Tree forecaster (baseline implementation)."""

import numpy as np

from aeon.forecasting.setar._setar import SETARForecaster


class SETARTreeForecaster:
    """Baseline global SETAR-Tree forecaster."""

    def __init__(self, lags=1, threshold_lag=1):
        self.lags = lags
        self.threshold_lag = threshold_lag

    def fit(self, y):
        """
        Fit the SETAR-Tree forecaster.

        Parameters
        ----------
        y : list of 1D numpy arrays
            Collection of univariate time series.
        """
        self.models_ = []

        for series in y:
            model = SETARForecaster(
                lags=self.lags,
                threshold_lag=self.threshold_lag,
            )
            model.fit(series)
            self.models_.append(model)

        return self

    def predict(self, fh):
        """Generate forecasts by averaging SETAR predictions."""
        preds = np.array([m.predict(fh) for m in self.models_])
        return preds.mean(axis=0).ravel()
