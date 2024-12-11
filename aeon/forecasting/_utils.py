"""
Forecasting utilities class.

Contains useful utility methods for forecasting time series data.

"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def calc_seasonal_period(data):
    """Estimate the seasonal period based on the autocorrelation of the series."""
    lags = _acf(data, 24)
    lags = np.concatenate((np.array([1.0]), lags))
    peaks = []
    mean_lags = np.mean(lags)
    for i in range(1, len(lags) - 1):  # Skip the first (lag 0) and last elements
        if lags[i] >= lags[i - 1] and lags[i] >= lags[i + 1] and lags[i] > mean_lags:
            peaks.append(i)
    if not peaks:
        return 1
    else:
        return peaks[0]


@njit(cache=True, fastmath=True)
def _acf(X, max_lag):
    length = len(X)
    X_t = np.zeros(max_lag, dtype=float)
    for lag in range(1, max_lag + 1):
        lag_length = length - lag
        x1 = X[:-lag]
        x2 = X[lag:]
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
            X_t[lag - 1] = 1
        elif v1_is_zero or v2_is_zero:  # One zero variance
            # the other not
            X_t[lag - 1] = 0
        else:
            X_t[lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)
    return X_t
