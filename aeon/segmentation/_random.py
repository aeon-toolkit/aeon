"""RandomSegmenter class, randomly segments a series."""


import random

from aeon.segmentation.base import BaseSegmenter


class RandomSegmenter(BaseSegmenter):
    """Dummy Segmenter.

    Randomly segments a time series.
    """

    _tags = {
        "X_inner_type": "ndarray",
        "capability:missing_values": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(self, random_state=None, n_segments=2):
        self.random_state = random_state
        self.breakpoints_ = []
        super(RandomSegmenter, self).__init__(n_segments=n_segments, axis=1)

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
        return self.breakpoints_
