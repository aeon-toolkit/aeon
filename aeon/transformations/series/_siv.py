"""Recursive Median Sieve filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["SIVSeriesTransformer"]


import numpy as np
from scipy.ndimage import median_filter

from aeon.transformations.series.base import BaseSeriesTransformer


class SIVSeriesTransformer(BaseSeriesTransformer):
    """Filter a times series using Recursive Median Sieve (SIV).

    Parameters
    ----------
    window_length : list of int or int, default=[3, 5, 7]
        The filter windows lenths (recommended increasing value).

    Notes
    -----
    Use scipy.ndimage.median_filter instead of scipy.signal.medfilt :
    The more general function scipy.ndimage.median_filter has a more efficient
    implementation of a median filter and therefore runs much faster.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html

    References
    ----------
    .. [1] Bangham J. A. (1988).
       Data-sieving hydrophobicity plots.
       Analytical biochemistry, 174(1), 142–145.
       https://doi.org/10.1016/0003-2697(88)90528-3
    .. [2] Yli-Harja, O., Koivisto, P., Bangham, J. A., Cawley, G.,
       Harvey, R., & Shmulevich, I. (2001).
       Simplified implementation of the recursive median sieve.
       Signal Process., 81(7), 1565–1570.
       https://doi.org/10.1016/S0165-1684(01)00054-8

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._siv import SIVSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> siv = SIVSeriesTransformer()
    >>> X_ = siv.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, window_length=None):
        self.window_length = window_length
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
        window_length = self.window_length
        if window_length is None:
            window_length = [3, 5, 7]
        if not isinstance(window_length, list):
            window_length = [window_length]

        # Compute SIV
        X_ = X

        for w in window_length:
            footprint = np.ones((1, w))
            X_ = median_filter(X_, footprint=footprint)

        return X_
