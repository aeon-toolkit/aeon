"""DummySegmentation class, randomly segments."""


import random

from aeon.segmentation import BaseSegmenter


class DummySegmenter(BaseSegmenter):
    """Dummy Segmenter."""

    _tags = {
        "capability:missing_values": True,
        "capability:multivariate": True,
    }

    def __init__(self, random_state=None, n_segments=None):
        self.random_state = random_state
        super(DummySegmenter, self).__init__(n_segments=n_segments, axis=1)

    def _fit(self, X, y=None):
        """Fit the dummy segmenter.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_channels, series_length]
        y : array-like, shape = [n_instances] - the class labels

        Returns
        -------
        self : reference to self.
        """
        self.n_segments_ = self.n_segments
        if self.n_segments_ is None:
            self.n_segments_ = 2
        length = len(X)
        self.breakpoints_ = random.sample(range(0, length), self.n_segments_ - 1)
        return self

    def _predict(self, X):
        """Generate breakpoints."""
        return self.breakpoints_
