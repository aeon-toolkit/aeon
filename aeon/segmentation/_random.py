"""RandomSegmenter class, randomly segments a series."""


import random

import numpy as np

from aeon.segmentation.base import BaseSegmenter


class RandomSegmenter(BaseSegmenter):
    """Random Segmenter.

    Randomly segments a time series.
    """

    _tags = {
        "capability:missing_values": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
    }

    def __init__(self, random_state=None, n_segments=2):
        self.random_state = random_state
        self.breakpoints_ = []
        super().__init__(n_segments=n_segments, axis=1)

    def _fit(self, X, y=None):
        """Fit the dummy segmenter.

        Parameters
        ----------
        X : 2D np.ndarray
            Time series of shape `(n_channels, series_length)`
        y : np.ndarray or None, default = None

        Returns
        -------
        self : reference to self.
        """
        length = X.shape[1]

        rng = random.Random()
        rng.seed(self.random_state)

        points = rng.sample(range(0, length), self.n_segments - 1)
        self.breakpoints_ = sorted(points)
        return self

    def _predict(self, X):
        """Generate breakpoints."""
        return np.array(self.breakpoints_)

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
