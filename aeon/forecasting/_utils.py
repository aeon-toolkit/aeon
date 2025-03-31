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


def kpss_test(y, regression="ct", lags=None):  # Test if time series is stationary
    """
    Implement the KPSS test for stationarity.

    Parameters
    ----------
    y (array-like): Time series data
    regression (str): 'c' for constant, 'ct' for constant + trend
    lags (int): Number of lags for HAC variance estimation (default: sqrt(n))

    Returns
    -------
    kpss_stat (float): KPSS test statistic
    stationary (bool): Whether the series is stationary according to the test
    """
    y = np.asarray(y)
    n = len(y)

    # Step 1: Fit regression model to estimate residuals
    if regression == "c":  # Constant
        X = np.ones((n, 1))
    elif regression == "ct":  # Constant + Trend
        X = np.column_stack((np.ones(n), np.arange(1, n + 1)))
    else:
        raise ValueError("regression must be 'c' or 'ct'")

    beta = np.linalg.inv(X.T @ X) @ X.T @ y  # Estimate regression coefficients
    residuals = y - X @ beta  # Get residuals (u_t)

    # Step 2: Compute cumulative sum of residuals (S_t)
    S_t = np.cumsum(residuals)

    # Step 3: Estimate long-run variance (HAC variance)
    if lags is None:
        lags = int(np.sqrt(n))  # Default lag length

    gamma_0 = np.mean(residuals**2)  # Lag-0 autocovariance
    gamma = [np.sum(residuals[k:] * residuals[:-k]) / n for k in range(1, lags + 1)]

    # Bartlett weights
    weights = [1 - (k / (lags + 1)) for k in range(1, lags + 1)]

    # Long-run variance
    sigma_squared = gamma_0 + 2 * np.sum([w * g for w, g in zip(weights, gamma)])

    # Step 4: Calculate the KPSS statistic
    kpss_stat = np.sum(S_t**2) / (n**2 * sigma_squared)

    if regression == "ct":
        # p. 162 Kwiatkowski et al. (1992): y_t = beta * t + r_t + e_t,
        # where beta is the trend, r_t a random walk and e_t a stationary
        # error term.
        crit = 0.146
    else:  # hypo == "c"
        # special case of the model above, where beta = 0 (so the null
        # hypothesis is that the data is stationary around r_0).
        crit = 0.463

    return kpss_stat, kpss_stat < crit
