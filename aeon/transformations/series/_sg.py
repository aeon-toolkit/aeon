"""Savitzky-Golay filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["SGSeriesTransformer"]


from scipy.signal import savgol_filter

from aeon.transformations.series.base import BaseSeriesTransformer


class SGSeriesTransformer(BaseSeriesTransformer):
    """Filter a times series using Savitzky-Golay (SG).

    Parameters
    ----------
    window_length : int, default=5
        The length of the filter window (i.e., the number of coefficients).
        window_length must be less than or equal to the size of the input.

    polyorder : int, default=2
        The order of the polynomial used to fit the samples.
        polyorder must be less than window_length.


    Notes
    -----
    More information of the SciPy savgol_filter function used
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    References
    ----------
    .. [1] Savitzky, A., & Golay, M. J. (1964).
       Smoothing and differentiation of data by simplified least squares procedures.
       Analytical chemistry, 36(8), 1627-1639.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._sg import SGSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> sg = SGSeriesTransformer()
    >>> X_ = sg.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : np.ndarray
            time series in shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility

        Returns
        -------
        transformed version of X
        """
        # Compute SG
        X_ = savgol_filter(X, self.window_length, self.polyorder)

        return X_
