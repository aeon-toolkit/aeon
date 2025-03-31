"""Sliding Window transformation."""

__maintainer__ = []
__all__ = ["SlidingWindowTransformer"]

import numpy as np

from aeon.transformations.format.base import BaseFormatTransformer


class SlidingWindowTransformer(BaseFormatTransformer):
    """
    Create windowed views of a series by extracting fixed-length overlapping segments.

    This transformer generates multiple subsequences (windows) of a specified width from
    the input time series. Each window represents a shifted view of the series, moving
    forward by one time step.

    Parameters
    ----------
    window_size : int, optional (default=100)
        The number of consecutive time points in each window.

    Notes
    -----
    - The function assumes that `window_width` is smaller than the length of `series`.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.format import SlidingWindowTransformer
    >>> X = np.array([1, 2, 3, 4, 5, 6])
    >>> transformer = SlidingWindowTransformer(3)
    >>> Xt = transformer.fit_transform(X)
    >>> print(Xt)
    ([[1, 2], [2, 3], [3, 4], [4, 5]], [3, 4, 5, 6], [0, 1, 2, 3])


    Returns
    -------
    X : np.ndarray (2D)
        A numpy array where each element is a window (subsequence) of length
        `window_width - 1` from the original series.
    Y : np.ndarray (1D)
        A numpy array containing the next value in the series for each window.
    indices : list of int
        A list of starting indices corresponding to each extracted window.

    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
        "output_data_type": "Tuple",
    }

    def __init__(self, window_size: int = 100):
        super().__init__(axis=1)
        if window_size <= 1:
            raise ValueError(f"window_size must be > 1, got {window_size}")
        self.window_size = window_size

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            The input time series from which windows will be created.
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt: 2D np.ndarray
            transformed version of X
        """
        X = X[0]
        # Generate windowed versions of train and test sets
        X_t = np.zeros((len(X) - self.window_size + 1, self.window_size - 1))
        Y_t = np.zeros(len(X) - self.window_size + 1)
        indices = np.zeros(len(X) - self.window_size + 1)
        for i in range(len(X) - self.window_size + 1):
            X_t[i] = X[
                i : i + self.window_size - 1
            ]  # Create a view from current index onward
            Y_t[i] = X[i + self.window_size - 1]  # Next value
            indices[i] = i
        return X_t, Y_t, indices
