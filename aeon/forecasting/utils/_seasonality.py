"""Seasonality Tools.

Includes autocorrelation function (ACF) and seasonal period estimation.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def acf(X, max_lag):
    """
    Compute the sample autocorrelation function (ACF) of a time series.

    Up to a specified maximum lag.

    The autocorrelation at lag k is defined as the Pearson correlation
    coefficient between the series and a lagged version of itself.
    If both segments at a given lag have zero variance, the function
    returns 1 for that lag. If only one segment has zero variance,
    the function returns 0.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        The input time series data.
    max_lag : int
        The maximum lag (number of steps) for which to
        compute the autocorrelation.

    Returns
    -------
    acf_values : np.ndarray, shape (max_lag,)
        The autocorrelation values for lags 1 through `max_lag`.

    Notes
    -----
    The function handles cases where the lagged segments have zero
    variance to avoid division by zero.
    The returned values correspond to
    lags 1, 2, ..., `max_lag` (not including lag 0).
    """
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


@njit(cache=True, fastmath=True)
def calc_seasonal_period(data):
    """
    Estimate the seasonal period of a time series using autocorrelation analysis.

    This function computes the autocorrelation function (ACF) of
    the input series up to lag 24. It then identifies peaks in the
    ACF above the mean value, treating the first such peak
    as the estimated seasonal period. If no peak is found,
    a period of 1 is returned.

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        The input time series data.

    Returns
    -------
    period : int
        The estimated seasonal period (lag) of the series. Returns 1 if no significant
        peak is detected in the autocorrelation.
    """
    lags = acf(data, 24)
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
