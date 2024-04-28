"""Mock segmenters for testing."""

import numpy as np

from aeon.segmentation import BaseSegmenter


class MockSegmenter(BaseSegmenter):
    """Mock segmenter for testing."""

    def __init__(self):
        super().__init__()

    _tags = {
        "capability:missing_values": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
    }

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """Generate breakpoints."""
        return np.array([1])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {}


class SupervisedMockSegmenter(MockSegmenter):
    """Mock segmenter for testing."""

    _tags = {
        "requires_y": True,
    }

    def _predict(self, X):
        """Generate breakpoints."""
        return np.array([1])
