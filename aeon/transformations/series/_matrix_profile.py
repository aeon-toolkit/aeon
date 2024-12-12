"""Implements matrix profile transformation."""

__maintainer__ = []
__all__ = ["MatrixProfileSeriesTransformer"]

from aeon.transformations.series.base import BaseSeriesTransformer


class MatrixProfileSeriesTransformer(BaseSeriesTransformer):
    """Calculate the matrix profile of a time series.

    Takes as input a single time series dataset and returns the matrix profile
    for that time series dataset. The matrix profile is a vector that stores the
    z-normalized Euclidean distance between any subsequence within a
    time series and its nearest neighbour.

    For more information on the matrix profile, see `th stumpy tutorial
    <https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html>`_

    Parameters
    ----------
    window_length : int
        Length of the sliding winodw for the matrix profile calculation.

    Notes
    -----
    Provides wrapper around functionality in `stumpy.stump
    <https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stump>`_

    Examples
    --------
    >>> from aeon.transformations.series import MatrixProfileSeriesTransformer
    >>> import numpy as np
    >>> series = np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1])  # doctest: +SKIP
    >>> transformer = MatrixProfileSeriesTransformer(window_length=4)  # doctest: +SKIP
    >>> mp = transformer.fit_transform(series)  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
        "python_dependencies": "stumpy",
    }

    def __init__(self, window_length=3):
        self.window_length = window_length
        self.matrix_profile_ = None
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray
            1D transformed version of X
            Matrix Profile of time series as output with length as
            (n_timepoints-window_length+1)
        """
        import stumpy

        X = X.squeeze()

        matrix_profile = stumpy.stump(X, self.window_length)
        return matrix_profile[:, 0].astype("float")
