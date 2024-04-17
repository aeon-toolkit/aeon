"""Auto-correlation transformations.

Module :mod:`aeon.transformations` implements auto-correlation
transformers.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["AutoCorrelationTransformer"]

import numpy as np
from numba import njit

from aeon.transformations.series.base import BaseSeriesTransformer


class AutoCorrelationTransformer(BaseSeriesTransformer):
    """Auto-correlation transformer.

    The autocorrelation function (ACF) measures how correlated a time series is
    with itself at different lags. The AutocorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified. This transformer
    intentionally uses a simple implementation without use of FFT and makes minimal
    adjustments to the ACF. It does not adjust for the mean or variance or trend.

    Parameters
    ----------
    adjusted : bool, default=False
        If True, then denominators for autocovariance are n-k, otherwise n.

    n_lags : int, default=None
        Number of lags to return autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs - 1).


    Examples
    --------
    >>> from aeon.transformations.acf import AutoCorrelationTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = AutoCorrelationTransformer(n_lags=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        n_lags=None,
    ):
        self.n_lags = n_lags
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # statsmodels acf function uses min(10 * np.log10(nobs), nobs - 1)
        if self.n_lags is None:
            self._n_lags = min(int(10 * np.log10(X.shape[1])), X.shape[1] - 1)
        else:
            self._n_lags = self.n_lags

        if X.shape[1] - self._n_lags < 3:
            raise ValueError(
                f"The number of lags is too large for the length of the "
                f"series, autocorrelation will be calculated "
                f"{X.shape[1]-self._n_lags} points."
            )
        return self._acf(X, max_lag=self._n_lags)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _acf(X, max_lag):
        n_channels, length = X.shape
        X_t = np.zeros((n_channels, max_lag))
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
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [{}, {"n_lags": 1}]
