"""Mock classifiers useful for testing and debugging.

Used in tests for the classifier base class.
"""

import numpy as np

from aeon.classification import BaseClassifier


class MockClassifier(BaseClassifier):
    """Dummy classifier for testing base class fit/predict."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))


class MockClassifierPredictProba(MockClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    def _predict_proba(self, X):
        """Predict proba dummy."""
        pred = np.zeros(shape=(len(X), 2))
        pred[:, 0] = 1
        return pred


class MockClassifierFullTags(MockClassifierPredictProba):
    """Dummy classifier able to handle all input types."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_type": ["np-list", "numpy3D"],
    }


class MockClassifierMultiTestParams(BaseClassifier):
    """Dummy classifier for testing base class fit/predict with multiple test params.

    Parameters
    ----------
    return_ones : bool, default=False
        If True, predict ones, else zeros.
    """

    def __init__(self, return_ones=False):
        self.return_ones = return_ones
        super().__init__()

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return [{"return_ones": False}, {"return_ones": True}]
