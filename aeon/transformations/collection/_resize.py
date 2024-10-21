"""Time series resizer."""

__all__ = ["Resizer"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Resizer(BaseCollectionTransformer):
    """Time series interpolator/re-sampler.

    Transformer that resizes series using np.linspace  is fitted on each channel
    independently. After transformation the collection will be a numpy array shape (
    n_cases, n_channels, length). It is not capable of sensibly handling missing
    values.

    Parameters
    ----------
    length : integer, the length of time series to resize to.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection import Resizer
    >>> # Unequal length collection of time series
    >>> X_list = []
    >>> for i in range(10): X_list.append(np.random.rand(5,10+i))
    >>> # Equal length collection of time series
    >>> X_array = np.random.rand(10,3,30)
    >>> trans = Resizer(length = 50)
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
        """Fit a linear function on each channel of each series, then resample.

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
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {"length": 10}
        return params
