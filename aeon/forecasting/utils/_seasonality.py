"""Seasonality Tools.

Includes autocorrelation function (ACF) and seasonal period estimation that uses ACF.
"""

import numpy as np
from numba import njit

__all__ = ["acf", "acf_seasonal_test", "calc_seasonal_period"]


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


@njit(cache=True, fastmath={"contract"})
def acf_seasonal_test(x, seasonal_period, threshold):
    """Test the autocorrelation at the seasonal lag for significance.

    Uses the standard sample ACF, which centres on the mean of the whole series
    and normalises by the variance of the whole series, unlike :func:`acf`. The
    ACF at ``seasonal_period`` is compared to its Bartlett standard error.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        The series to test, for example the raw or the differenced series.
    seasonal_period : int
        The lag to test.
    threshold : float
        Normal quantile the absolute standardised ACF must exceed.

    Returns
    -------
    bool
        ``True`` if the ACF at ``seasonal_period`` is significant. ``False`` if
        the series is too short or has no variance.
    """
    n = x.shape[0]
    m = seasonal_period
    if n <= m:
        return False

    mean_x = 0.0
    for i in range(n):
        mean_x += x[i]
    mean_x /= n

    centred = np.empty(n, dtype=np.float64)
    denom = 0.0
    for i in range(n):
        c = x[i] - mean_x
        centred[i] = c
        denom += c * c
    if denom <= 1e-12:
        return False

    prior_sq = 0.0
    acf_at_m = 0.0
    for lag in range(1, m + 1):
        s = 0.0
        for i in range(n - lag):
            s += centred[i] * centred[i + lag]
        r = s / denom
        if lag < m:
            prior_sq += r * r
        else:
            acf_at_m = r

    se_sq = (1.0 + 2.0 * prior_sq) / n
    if se_sq <= 0.0 or not np.isfinite(se_sq):
        return False
    return abs(acf_at_m) / se_sq**0.5 > threshold


def _seasonal_forecast(seasonal_factors, h, seasonal_period, n_train):
    """Repeat the learned seasonal cycle in phase for ``h`` future steps."""
    return seasonal_factors[(int(n_train) + np.arange(int(h))) % int(seasonal_period)]


@njit(cache=True, fastmath={"contract"})
def _seasonal_decompose_numba(y, seasonal_period, requested_mult, mult_floor):
    """Classical seasonal decomposition with centred-MA detrending in one pass.

    1. Estimate a trend via the centred moving average of period
       ``seasonal_period`` (odd ``m``: uniform window; even ``m``: 2*m smoothing
       with ``[0.5, 1, ..., 1, 0.5] / m`` weights).
    2. Detrend additively or multiplicatively.
    3. Estimate seasonal indices as the per-position mean of the detrended
       series, ignoring positions where the MA is undefined.
    4. Normalise (sum-zero for additive, mean-one for multiplicative).

    Multiplicative falls back to additive when ``y`` contains non-positive
    values, when the MA contains non-positive values, or when factors are
    non-finite or below ``mult_floor``.

    Returns
    -------
    type_code : int
        ``0`` for "none" (no decomposition applied), ``1`` for additive,
        ``2`` for multiplicative.
    factors : np.ndarray
        Seasonal factors of length ``seasonal_period``. All zeros when
        ``type_code == 0``; the caller should ignore the array in that case.
    adjusted : np.ndarray
        Seasonally adjusted series of the same length as ``y``. Equal to a
        copy of ``y`` when ``type_code == 0``.
    """
    n = y.shape[0]
    m = seasonal_period
    half = m // 2

    factors = np.zeros(m, dtype=np.float64)

    # Centred MA requires at least m points (odd m) or m + 1 points (even m).
    if m % 2 == 1:
        if n < m:
            return 0, factors, y.copy()
    else:
        if n < m + 1:
            return 0, factors, y.copy()

    inv_m = 1.0 / m
    ma = np.empty(n, dtype=np.float64)
    for i in range(n):
        ma[i] = np.nan

    if m % 2 == 1:
        for t in range(half, n - half):
            s = 0.0
            for k in range(t - half, t + half + 1):
                s += y[k]
            ma[t] = s * inv_m
    else:
        for t in range(half, n - half):
            s = 0.5 * (y[t - half] + y[t + half])
            for k in range(t - half + 1, t + half):
                s += y[k]
            ma[t] = s * inv_m

    use_mult = requested_mult
    if use_mult:
        for i in range(n):
            if y[i] <= 0.0:
                use_mult = False
                break
    if use_mult:
        for t in range(half, n - half):
            if ma[t] <= 0.0:
                use_mult = False
                break

    sums = np.empty(m, dtype=np.float64)
    counts = np.empty(m, dtype=np.int64)

    if use_mult:
        for p in range(m):
            sums[p] = 0.0
            counts[p] = 0
        for t in range(half, n - half):
            d = y[t] / ma[t]
            if np.isfinite(d):
                pos = t % m
                sums[pos] += d
                counts[pos] += 1

        valid = True
        for p in range(m):
            if counts[p] == 0:
                valid = False
                break
            factors[p] = sums[p] / counts[p]

        if valid:
            factor_sum = 0.0
            for p in range(m):
                factor_sum += factors[p]
            factor_mean = factor_sum / m
            if np.isfinite(factor_mean) and factor_mean > mult_floor:
                ok = True
                min_factor = 1e300
                for p in range(m):
                    f = factors[p] / factor_mean
                    factors[p] = f
                    if not np.isfinite(f):
                        ok = False
                        break
                    if f < min_factor:
                        min_factor = f
                if ok and min_factor >= mult_floor:
                    adjusted = np.empty(n, dtype=np.float64)
                    all_finite = True
                    for i in range(n):
                        rep = factors[i % m]
                        v = y[i] / rep
                        adjusted[i] = v
                        if not np.isfinite(v):
                            all_finite = False
                            break
                    if all_finite:
                        return 2, factors, adjusted

    for p in range(m):
        sums[p] = 0.0
        counts[p] = 0
    for t in range(half, n - half):
        d = y[t] - ma[t]
        if np.isfinite(d):
            pos = t % m
            sums[pos] += d
            counts[pos] += 1

    for p in range(m):
        if counts[p] == 0:
            zeros = np.zeros(m, dtype=np.float64)
            return 0, zeros, y.copy()
        factors[p] = sums[p] / counts[p]

    factor_sum = 0.0
    for p in range(m):
        factor_sum += factors[p]
    factor_mean = factor_sum / m
    for p in range(m):
        factors[p] -= factor_mean
        if not np.isfinite(factors[p]):
            zeros = np.zeros(m, dtype=np.float64)
            return 0, zeros, y.copy()

    adjusted = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = y[i] - factors[i % m]
        adjusted[i] = v
        if not np.isfinite(v):
            zeros = np.zeros(m, dtype=np.float64)
            return 0, zeros, y.copy()
    return 1, factors, adjusted
