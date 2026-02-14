"""Seasonality Tools.

Includes autocorrelation function (ACF) and seasonal period estimation that uses ACF.
"""

import numpy as np
from numba import njit

__all__ = ["acf", "calc_seasonal_period"]


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
    length = X.shape[0]
    acf_function = np.empty(max_lag, dtype=np.float64)
    eps = 1e-9

    for lag in range(1, max_lag + 1):
        lag_length = length - lag

        s1 = 0.0
        s2 = 0.0
        ss1 = 0.0
        ss2 = 0.0
        s12 = 0.0

        for i in range(lag_length):
            a = X[i]
            b = X[i + lag]
            s1 += a
            s2 += b
            ss1 += a * a
            ss2 += b * b
            s12 += a * b

        v1 = ss1 - (s1 * s1) / lag_length
        v2 = ss2 - (s2 * s2) / lag_length

        if v1 <= eps and v2 <= eps:
            acf_function[lag - 1] = 1.0
        elif v1 <= eps or v2 <= eps:
            acf_function[lag - 1] = 0.0
        else:
            cov = s12 - (s1 * s2) / lag_length
            acf_function[lag - 1] = cov / np.sqrt(v1 * v2)

    return acf_function


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
    lags = acf(data, min(24, len(data) - 3))
    lags = np.concatenate((np.array([1.0]), lags))
    mean_lags = np.mean(lags)
    for i in range(1, len(lags) - 1):  # Skip the first (lag 0) and last elements
        if lags[i] >= lags[i - 1] and lags[i] >= lags[i + 1] and lags[i] > mean_lags:
            return i
    return 1
