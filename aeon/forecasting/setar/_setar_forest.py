"""SETAR Forest forecaster."""

import numpy as np

from aeon.forecasting.setar._setar_tree import SETARTreeForecaster


class SETARForestForecaster:
    """Bootstrap ensemble of SETAR tree forecasters."""

    def __init__(
        self,
        n_estimators: int = 10,
        lags: int = 1,
        threshold_lag: int = 1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.lags = lags
        self.threshold_lag = threshold_lag
        self.random_state = random_state

    def fit(self, y):
        """Fit an ensemble of SETAR trees."""
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for _ in range(self.n_estimators):
            indices = rng.integers(0, len(y), size=len(y))
            y_boot = [y[i] for i in indices]

            tree = SETARTreeForecaster(lags=self.lags, threshold_lag=self.threshold_lag)
            tree.fit(y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, fh):
        """Average predictions across trees."""
        preds = np.asarray([tree.predict(fh) for tree in self.trees_])
        return preds.mean(axis=0)
