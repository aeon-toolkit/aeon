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
