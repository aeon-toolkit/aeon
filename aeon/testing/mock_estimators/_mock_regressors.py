"""Mock regressors useful for testing and debugging."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "MockRegressor",
    "MockRegressorFullTags",
]

from sklearn.utils import check_random_state

from aeon.regression.base import BaseRegressor


class MockRegressor(BaseRegressor):
    """Dummy regressor for testing base class fit/predict."""

    def __init__(self, random_state=None):
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        rng = check_random_state(self.random_state)
        return rng.random(size=(len(X)))


class MockRegressorFullTags(BaseRegressor):
    """Dummy regressor for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(self, random_state=None):
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        rng = check_random_state(self.random_state)
        return rng.random(size=(len(X)))
