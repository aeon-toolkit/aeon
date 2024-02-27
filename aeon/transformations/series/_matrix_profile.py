"""Implements matrix profile transformation."""

__maintainer__ = []
__all__ = ["MatrixProfileSeriesTransformer"]

from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies


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
        super().__init__()

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
        _check_soft_dependencies("stumpy", severity="error")
        import stumpy

        self.matrix_profile_ = stumpy.stump(X, self.window_length)
        return self.matrix_profile_[:, 0].astype("float")

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
