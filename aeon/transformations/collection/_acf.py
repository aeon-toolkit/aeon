"""Autocorrelation function transformer.

Efficient implementation for collections using numba.
"""

__maintainer__ = []
__all__ = ["AutocorrelationFunctionTransformer"]

import numpy as np
from numba import njit

from aeon.transformations.collection.base import BaseCollectionTransformer


class AutocorrelationFunctionTransformer(BaseCollectionTransformer):
    """Autocorrelation function transformer.

    The autocorrelation function measures how correlated a timeseries is
    with itself at different lags. The AutocorrelationFunctionTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Efficient implementation for collections using numba

    Parameters
    ----------
    n_lags : int, None or callable, default=None
        The maximum number of autocorrelation terms to use. If None, set to
        n_timepoints/4. If callable, the
        function should take a 3D numpy array of shape (n_cases, n_channels,
        n_timepoints) and return an integer.
    min_values : int, default=0
        Never use fewer than this number of terms to find a correlation unless the
        series length is too short. This will reduce n_lags if needed.

    Examples
    --------
    >>> from aeon.transformations.collection import AutocorrelationFunctionTransformer
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X = make_example_3d_numpy(n_cases=4, n_channels=2, n_timepoints=20,
    ...                           random_state=0, return_y=False)
    >>> tnf = AutocorrelationFunctionTransformer(n_lags=10)
    >>> tnf.fit(X)
    AutocorrelationFunctionTransformer(...)
    >>> print(tnf.transform(X)[0])
    [[ 0.10642255 -0.04497476 -0.27607675 -0.24169331  0.04717655  0.07221666
      -0.36798515 -0.53768553  0.07550288  0.08557519]
     [-0.21166957  0.24992846 -0.38036068  0.10243325 -0.18565336  0.05619381
      -0.19569665  0.28835692 -0.42359509  0.21378191]]
    """

    _tags = {
        "fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_lags=None,
        min_values=0,
    ):
        self.n_lags = n_lags
        self.min_values = min_values

        super().__init__()

    def _transform(self, X, y=None):
        n_cases, n_channels, n_timepoints = X.shape
        if self.n_lags is None:
            lags = n_timepoints / 4
        else:
            lags = self.n_lags(X) if callable(self.n_lags) else self.n_lags
        if lags > n_timepoints - self.min_values:
            lags = n_timepoints - self.min_values
        if lags < 0:
            lags = 1

        if lags > n_timepoints - 1:
            raise ValueError(
                f"lags ({lags}) must be smaller than n_timepoints - 1 "
                f"({n_timepoints - 1})."
            )

        Xt = np.zeros((n_cases, n_channels, lags))
        for n in range(n_channels):
            Xt[:, n, :] = self._acf_2d(X[:, n, :], lags)

        return Xt

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _acf_2d(X, max_lag):
        n_cases, length = X.shape

        X_t = np.zeros((n_cases, max_lag))
        for i, x in enumerate(X):
            for lag in range(1, max_lag + 1):
                lag_length = length - lag
                x1, x2 = x[:-lag], x[lag:]
                s1 = np.sum(x1)
                s2 = np.sum(x2)
                m1 = s1 / lag_length
                m2 = s2 / lag_length
                ss1 = np.sum(x1 * x1)
                ss2 = np.sum(x2 * x2)
                v1 = ss1 - s1 * m1
                v2 = ss2 - s2 * m2
                v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
                if v1_is_zero and v2_is_zero:  # Both zero variance,
                    # so must be 100% correlated
                    X_t[i][lag - 1] = 1
                elif v1_is_zero or v2_is_zero:  # One zero variance
                    # the other not
                    X_t[i][lag - 1] = 0
                else:
                    X_t[i][lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)

        return X_t

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
        return {
            "n_lags": 4,
        }
