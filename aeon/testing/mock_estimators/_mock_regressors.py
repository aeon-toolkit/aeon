"""Mock regressor useful for testing and debugging.

Used in tests for the regressor base class.
"""

import numpy as np

from aeon.regression.base import BaseRegressor


class MockRegressor(BaseRegressor):
    """Dummy regressor for testing base class fit/predict."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.random.random(size=(len(X)))


class MockHandlesAllInput(BaseRegressor):
    """Dummy regressor for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.random.random(size=(len(X)))
