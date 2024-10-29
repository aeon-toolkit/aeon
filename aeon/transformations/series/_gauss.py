"""Gaussian filter transformation."""

__maintainer__ = ["Cyril-Meyer"]
__all__ = ["GaussSeriesTransformer"]


from scipy.ndimage import gaussian_filter1d

from aeon.transformations.series.base import BaseSeriesTransformer


class GaussSeriesTransformer(BaseSeriesTransformer):
    """Filter a times series using Gaussian filter.

    Parameters
    ----------
    sigma : float, default=1
        Standard deviation for the Gaussian kernel.

    order : int, default=0
        An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a
        Gaussian.


    Notes
    -----
    More information of the SciPy gaussian_filter1d function used
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    References
    ----------
    .. [1] Rafael C. Gonzales and Paul Wintz. 1987.
       Digital image processing.
       Addison-Wesley Longman Publishing Co., Inc., USA.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._gauss import GaussSeriesTransformer
    >>> X = np.random.random((2, 100)) # Random series length 100
    >>> gauss = GaussSeriesTransformer(sigma=5)
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
        # Compute Gaussian filter
        X_ = gaussian_filter1d(X, self.sigma, self.axis, self.order)

        return X_
