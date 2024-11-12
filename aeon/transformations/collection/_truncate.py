"""Truncation transformer - truncate unequal length collections."""

__all__ = ["Truncator"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Truncator(BaseCollectionTransformer):
    """Truncate unequal length time series to a lower bounds.

    Truncates all series in collection between lower/upper range bounds. This
    transformer assumes that all series have the same number of channels (dimensions)
    and that all channels in a single series are the same length.

    Parameters
    ----------
    truncated_length : int, default=None
        bottom range of the values to truncate can also be used to truncate
        to a specific length.
        if None, will find the shortest sequence and use instead.

    Examples
    --------
    >>> from aeon.transformations.collection import Truncator
    >>> import numpy as np
    >>> X = []
    >>> for i in range(10): X.append(np.random.random((4, 75 + i)))
    >>> truncator = Truncator(truncated_length=10)
    >>> X2 = truncator.fit_transform(X)
    >>> X2.shape
    (10, 4, 10)

    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "removes_unequal_length": True,
    }

    def __init__(self, truncated_length=None):
        self.truncated_length = truncated_length
        super().__init__()

    @staticmethod
    def _get_min_length(X):
        min_length = X[0].shape[1]
        for x in X:
            if x.shape[1] < min_length:
                min_length = x.shape[1]

        return min_length

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series or 3D numpy of equal length
            series
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        # If lower is none, set to the minimum length in X
        min_length = self._get_min_length(X)
        if self.truncated_length is None:
            self.truncated_length_ = min_length
        elif min_length < self.truncated_length:
            self.truncated_length_ = min_length
        else:
            self.truncated_length_ = self.truncated_length

        return self

    def _transform(self, X, y=None):
        """Truncate X and return a transformed version.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series.
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : numpy3D array (n_cases, n_channels, self.truncated_length_)
            truncated time series from X.
        """
        min_length = self._get_min_length(X)
        if min_length < self.truncated_length_:
            raise ValueError(
                "Error: min_length of series \
                    is less than the one found when fit or set."
            )
        Xt = np.array([x[:, : self.truncated_length_] for x in X])
        return Xt

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
        params = {"truncated_length": 5}
        return params
