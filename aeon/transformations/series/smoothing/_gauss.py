"""Gaussian filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["GaussianFilter"]


from scipy.ndimage import gaussian_filter1d

from aeon.transformations.series.base import BaseSeriesTransformer


class GaussianFilter(BaseSeriesTransformer):
    """Filter a time series using Gaussian filter.

    Wrapper for the SciPy ``gaussian_filter1d`` function.

    Parameters
    ----------
    sigma : float, default=1
        Standard deviation for the Gaussian kernel.
    order : int, default=0
        An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a
        Gaussian.

    References
    ----------
    .. [1] Chou, Y. L. "Statistical Analysis, Section 17.9." New York: Holt
        International (1975).

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series.smoothing import GaussianFilter
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> gauss = GaussianFilter(sigma=5)
    >>> X_ = gauss.fit_transform(X)
    >>> X_.shape
    (2, 100)
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, sigma=1, order=0):
        self.sigma = sigma
        self.order = order

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
        return gaussian_filter1d(X, self.sigma, axis=self.axis, order=self.order)
