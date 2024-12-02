"""Mock classifiers useful for testing and debugging."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "MockClassifier",
    "MockClassifierPredictProba",
    "MockClassifierFullTags",
    "MockClassifierParams",
    "MockClassifierComposite",
]

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier


class MockClassifier(BaseClassifier):
    """Mock classifier for testing fit/predict."""

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        self.foo_ = "bar"
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))


class MockClassifierPredictProba(MockClassifier):
    """Mock classifier for testing fit/predict/predict_proba."""

    def _predict_proba(self, X):
        """Predict proba dummy."""
        pred = np.zeros(shape=(len(X), 2))
        pred[:, 0] = 1
        return pred


class MockClassifierFullTags(MockClassifierPredictProba):
    """Mock classifier able to handle all input types."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }


class MockClassifierParams(MockClassifier):
    """Mock classifier for testing fit/predict with multiple parameters.

    Parameters
    ----------
    return_ones : bool, default=False
        If True, predict ones, else zeros.
    """

    def __init__(self, return_ones=False, value=50):
        self.return_ones = return_ones
        self.value = value
        super().__init__()

    def _predict(self, X):
        """Predict dummy."""
        return (
            np.zeros(shape=(len(X),))
            if not self.return_ones
            else np.ones(shape=(len(X),))
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return [{"return_ones": False, "value": 10}, {"return_ones": True}]


class MockClassifierComposite(BaseClassifier):
    """Mock classifier which contains another mock classfier."""

    def __init__(self, mock=None):
        self.mock = mock
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        self.mock_ = (
            MockClassifier().fit(X, y)
            if self.mock is None
            else _clone_estimator(self.mock).fit(X, y)
        )
        self.foo_ = "bar"
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self.mock_.predict(X)
