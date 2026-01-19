import numpy as np
from numpy.random import RandomState

from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.setar._setar_tree import SETARTreeForecaster


class SETARForestForecaster(BaseForecaster):
    """
    Ensemble of SETAR-Tree forecasters.
    """

    _tags = {
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "capability:global_forecasting": True,
    }

    def __init__(self, n_estimators=10, lags=1, threshold_lag=1, random_state=None):
        self.n_estimators = n_estimators
        self.lags = lags
        self.threshold_lag = threshold_lag
        self.random_state = random_state
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        rng = RandomState(self.random_state)
        self.trees_ = []

        for _ in range(self.n_estimators):
            indices = rng.choice(len(y), size=len(y), replace=True)
            y_boot = [y[i] for i in indices]

            tree = SETARTreeForecaster(
                lags=self.lags, threshold_lag=self.threshold_lag
            )
            tree.fit(y_boot)
            self.trees_.append(tree)

        return self

    def _predict(self, fh, X=None):
        preds = np.array([tree.predict(fh) for tree in self.trees_])
        return preds.mean(axis=0)
