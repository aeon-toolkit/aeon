"""Time series interpolator/re-sampler."""

__all__ = ["TSInterpolator"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


class TSInterpolator(BaseCollectionTransformer):
    """Time series interpolator/re-sampler.

    Transformer that rescales series for another number of points.
    For each series, np.interp  is fitted on each channel independently.
    After transformation each series will be a 2D numpy array (n_channels, length).

    Parameters
    ----------
    length : integer, the length of time series to resize to.

    Example
    -------
    >>> import numpy as np
    >>> from aeon.transformations.collection.interpolate import TSInterpolator
    >>> # Unequal length collection of time series
    >>> X_list = []
    >>> for i in range(10): X_list.append(np.random.rand(5,10+i))
    >>> # Equal length collection of time series
    >>> X_array = np.random.rand(10,3,30)
    >>> trans = TSInterpolator(length = 50)
    >>> X_new = trans.fit_transform(X_list)
    >>> X_new.shape
    (10, 5, 50)
    >>> X_new = trans.fit_transform(X_array)
    >>> X_new.shape
    (10, 3, 50)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(self, length):
        """Initialize estimator.

        Parameters
        ----------
        length : integer, the length of time series to resize to.
        """
        if length <= 0 or (not isinstance(length, int)):
            raise ValueError("resizing length must be integer and > 0")

        self.length = length
        super().__init__()

    def _transform(self, X, y=None):
        """Take series in each cell, train linear interpolation and samples n.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or
            list size [n_cases] of 2D nump arrays, case i has shape (n_channels,
            length_i). Collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        3D numpy array of shape (n_cases, n_channels, self.length)
        """
        Xt = []
        for x in X:
            x_new = np.zeros((x.shape[0], self.length))
            x2 = np.linspace(0, 1, x.shape[1])
            x3 = np.linspace(0, 1, self.length)
            for i, row in enumerate(x):
                x_new[i] = np.interp(x3, x2, row)
            Xt.append(x_new)
        return np.array(Xt)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        params = {"length": 10}
        return params
