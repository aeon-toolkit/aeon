"""Time series linear interpolation resizer for unequal length."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["Resizer"]

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.unequal_length._commons import (
    _get_max_length,
    _get_min_length,
)


class Resizer(BaseCollectionTransformer):
    """Resize unequal length time series to equal, fixed length.

    Resize the series using linear interpolation to either a fixed length or
    finds the max/min length series across all series and channels and resizes
    all series to that length.

    Parameters
    ----------
    resized_length  : int, "min" or "max", default="min"
        Length to resize the series to. If "min", will resize the transformed series
        to the shortest series seen in ``fit``. If "max", will resize to the longest
        series seen in ``fit``. If an integer, will resize to that length.
        Calling ``fit`` is not required if ``resized_length`` is an int.

    Examples
    --------
    >>> from aeon.transformations.collection.unequal_length import Resizer
    >>> import numpy as np
    >>> X = []
    >>> for i in range(10): X.append(np.random.random((4, 75 + i)))
    >>> resizer = Resizer(resized_length=100)
    >>> X2 = resizer.fit_transform(X)
    >>> X2.shape
    (10, 4, 100)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "removes_unequal_length": True,
    }

    def __init__(self, resized_length="max"):
        self.resized_length = resized_length

        super().__init__()

        self.set_tags(**{"fit_is_empty": isinstance(resized_length, int)})

    def _fit(self, X, y=None):
        if self.resized_length == "min":
            self._resized_length = _get_min_length(X)
        elif self.resized_length == "max":
            self._resized_length = _get_max_length(X)
        else:
            raise ValueError("resized_length must be 'min', 'max' or an integer.")

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
        length = (
            self.resized_length
            if isinstance(self.resized_length, int)
            else self._resized_length
        )

        Xt = []
        for x in X:
            x_new = np.zeros((x.shape[0], length))
            x2 = np.linspace(0, 1, x.shape[1])
            x3 = np.linspace(0, 1, length)
            for i, row in enumerate(x):
                x_new[i] = np.interp(x3, x2, row)
            Xt.append(x_new)
        return np.array(Xt)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"resized_length": 15}
