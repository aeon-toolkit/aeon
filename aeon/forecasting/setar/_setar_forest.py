"""Ensemble of SETAR-Tree forecasters."""

import numpy as np
from numpy.random import RandomState

from aeon.forecasting.setar._setar_tree import SETARTreeForecaster


class SETARForestForecaster:
    """Ensemble of SETAR-Tree forecasters."""

    def __init__(self, n_estimators=10, lags=1, threshold_lag=1, random_state=None):
        self.n_estimators = n_estimators
        self.lags = lags
        self.threshold_lag = threshold_lag
        self.random_state = random_state

    def fit(self, y):
        """
        Fit the SETAR-Forest forecaster.

        Parameters
        ----------
        y : list of 1D numpy arrays
            Collection of univariate time series.
        """
        rng = RandomState(self.random_state)
        self.trees_ = []

        for _ in range(self.n_estimators):
            indices = rng.choice(len(y), size=len(y), replace=True)
            y_boot = [y[i] for i in indices]

            tree = SETARTreeForecaster(
                lags=self.lags,
                threshold_lag=self.threshold_lag,
            )
            tree.fit(y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, fh):
        """Generate forecasts by averaging ensemble predictions."""
        preds = np.array([t.predict(fh) for t in self.trees_])
        return preds.mean(axis=0).ravel()
