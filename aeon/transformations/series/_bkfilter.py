"""Baxter-King bandpass filter transformation."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["BKFilter"]


import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class BKFilter(BaseSeriesTransformer):
    """Filter a times series using the Baxter-King filter.

    The Baxter-King filter from econometrics  that uses a band pass filter to
    removes the cycle component (seasonality) from the time series based on weighted
    moving average with specified weights. It removes high or low frequency
    patterns and returns a centred weighted moving average of the original series.


    Parameters
    ----------
    low : float
        Minimum period for oscillations. Baxter and King recommend a value of 6
        for quarterly data and 1.5 for annual data.

    high : float
        Maximum period for oscillations. BK recommend 32 for U.S. business cycle
        quarterly data and 8 for annual data.

    K : int
        Lead-lag length of the filter. Baxter and King suggest a truncation
        length of 12 for quarterly data and 3 for annual data.

    Notes
    -----
    Adapted from statsmodels implementation
    https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/filters/bk_filter.py

    References
    ----------
    Baxter, M. and R. G. King. "Measuring Business Cycles: Approximate
        Band-Pass Filters for Economic Time Series." *Review of Economics and
        Statistics*, 1999, 81(4), 575-593.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._bkfilter import BKFilter
    >>> X = np.random.random((1,100)) # Random series length 100
    >>> bk = BKFilter()
    >>> X2 = bk.fit_transform(X)
    >>> X2.shape
    (1, 76)
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(
        self,
        low=6,
        high=32,
        K=12,
    ):
        self.low = low
        self.high = high
        self.K = K
        super().__init__(axis=0)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            time series in shape (n_timepoints, n_channels)

        Returns
        -------
        transformed cyclical version of X
        """
        from scipy.signal import fftconvolve

        omega_1 = 2.0 * np.pi / self.high
        omega_2 = 2.0 * np.pi / self.low
        bweights = np.zeros(2 * self.K + 1)
        bweights[self.K] = (omega_2 - omega_1) / np.pi
        j = np.arange(1, int(self.K) + 1)
        weights = 1 / (np.pi * j) * (np.sin(omega_2 * j) - np.sin(omega_1 * j))
        bweights[self.K + j] = weights
        bweights[: self.K] = weights[::-1]
        bweights -= bweights.mean()
        if X.ndim == 2:
            bweights = bweights[:, None]
        XTr = fftconvolve(X, bweights, mode="valid")

        return XTr

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {"low": 6, "high": 24, "K": 12}
        return params
