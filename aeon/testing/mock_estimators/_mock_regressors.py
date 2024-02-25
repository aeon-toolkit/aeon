"""Mock regressor useful for testing and debugging.

Used in tests for the regressor base class.
"""

import numpy as np

from aeon.regression.base import BaseRegressor

class MockRegressor(BaseRegressor):
    """Dummy regressor for testing base class fit/predict."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        rng = np.random.default_rng(self.random_state)
        return rng.random(size=(len(X)))


class MockHandlesAllInput(BaseRegressor):
    """Dummy regressor for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(self, random_state=None):
        self.random_state = random_state

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        rng = np.random.default_rng(self.random_state)
        return rng.random(size=(len(X)))
