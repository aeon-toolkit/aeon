"""Loss functions for optimiser."""

import numpy as np
from numba import njit

from aeon.forecasting.utils._extract_paras import (
    _ets_is_damped,
    _extract_arma_params,
    _extract_ets_params,
)

LOG_2PI = 1.8378770664093453


@njit(cache=True, fastmath={"contract"})
def _is_stationary_ar(coefs):
    """Check that an AR polynomial has all its roots outside the unit circle.

    The polynomial is ``1 - coefs[0] z - ... - coefs[-1] z^k``. Uses the
    Levinson-Durbin step-down recursion: the polynomial is stationary
    if and only if every reflection coefficient has absolute value below one.
    """
    a = coefs.copy()
    for k in range(len(a), 0, -1):
        r = a[k - 1]
        denom = 1.0 - r * r
        # |r| < 1 can still give denom == 0 in floating point
        if not np.abs(r) < 1.0 or not denom > 0.0:
            return False
        if k > 1:
            prev = np.empty(k - 1)
            for j in range(k - 1):
                prev[j] = (a[j] + r * a[k - 2 - j]) / denom
            a = prev
    return True


@njit(cache=True, fastmath={"contract"})
def _arima_fit(params, data, model):
    """Calculate the AIC of an ARIMA model given the parameters.

    Returns ``inf`` unless the AR part is stationary and the MA part
    invertible. Outside that region the conditional sum of squares can be
    spuriously small, which made larger, non-invertible models look better.
    """
    formatted_params = _extract_arma_params(params, model)  # Extract parameters
    if not _is_stationary_ar(formatted_params[1, : model[1]]):
        return np.inf
    if not _is_stationary_ar(-formatted_params[2, : model[2]]):
        return np.inf

    # Initialize residuals
    n = len(data)
    residuals = np.zeros(n)
    c = formatted_params[0][0] if model[0] else 0
    p = model[1]
    q = model[2]
    for t in range(max(p, q), n):
        ar_term = 0.0
        max_ar = min(p, t)
        for j in range(max_ar):
            ar_term += formatted_params[1, j] * data[t - j - 1]
        ma_term = 0.0
        max_ma = min(q, t)
        for j in range(max_ma):
            ma_term += formatted_params[2, j] * residuals[t - j - 1]
        y_hat = c + ar_term + ma_term
        residuals[t] = data[t] - y_hat
    return _arima_css_aic(residuals, max(p, q), len(params))


@njit(cache=True, fastmath={"contract"})
def _arima_css_aic(residuals, start, num_params):
    """AIC of an ARMA model from its conditional sum of squares residuals.

    The first ``start = max(p, q)`` residuals are conditioned on and not used.
    The variance is estimated from the remaining residuals, but the Gaussian
    log likelihood is evaluated over all ``len(residuals)`` observations, so
    models of different order are compared on the same number of observations.
    Counting only the ``len(residuals) - start`` observations used made each
    extra lag lower the AIC by about as much as an extra parameter raises it,
    so model selection favoured overly large orders.
    """
    n = len(residuals)
    if n <= start:
        return np.inf
    sse = 0.0
    for i in range(start, n):
        sse += residuals[i] * residuals[i]
    variance = sse / (n - start)
    likelihood = n * (LOG_2PI + np.log(variance) + 1.0)
    return likelihood + 2 * num_params


EPS = 1e-6
LARGE_LOSS = np.inf


@njit(inline="always", cache=True)
def safe_div(num, den):
    if den < EPS:
        return num / EPS
    else:
        return num / den


@njit(inline="always", cache=True)
def _is_finite(value):
    return not (np.isnan(value) or np.isinf(value))


@njit(inline="always", cache=True)
def _ets_params_are_valid(alpha, beta, gamma, phi, trend_type, seasonality_type):
    if not _is_finite(alpha) or alpha < 0.0 or alpha >= 1.0:
        return False
    if trend_type != 0:
        if not _is_finite(beta) or beta < 0.0 or beta >= alpha:
            return False
        if not _is_finite(phi) or phi <= 0.0 or phi > 1.0:
            return False
    if seasonality_type != 0:
        if not _is_finite(gamma) or gamma < 0.0 or gamma >= 1.0 - alpha:
            return False
    return True


@njit(inline="always", cache=True)
def _ets_state_is_valid(
    error_type,
    trend_type,
    seasonality_type,
    level,
    trend,
    seasonality,
    fitted_value,
    error,
):
    if not (
        _is_finite(level)
        and _is_finite(trend)
        and _is_finite(seasonality)
        and _is_finite(fitted_value)
        and _is_finite(error)
    ):
        return False
    if error_type == 2 and fitted_value <= EPS:
        return False
    if trend_type == 2 and (level <= EPS or trend <= EPS):
        return False
    if seasonality_type == 2:
        if seasonality <= EPS or fitted_value <= EPS:
            return False
    return True


@njit(cache=True)
def _ets_forecast_cycle_is_valid(
    error_type,
    trend_type,
    seasonality_type,
    level,
    trend,
    seasonality,
    phi,
    n_timepoints,
    seasonal_period,
):
    if error_type != 2 and seasonality_type != 2:
        return True
    for horizon in range(1, seasonal_period + 1):
        if phi == 1.0:
            phi_h = horizon
        else:
            phi_h = phi * (1.0 - phi**horizon) / (1.0 - phi)
        seasonal_index = (n_timepoints + horizon - 1) % seasonal_period
        fitted_value, _, _ = _ets_predict_value(
            trend_type,
            seasonality_type,
            level,
            trend,
            seasonality[seasonal_index],
            phi_h,
        )
        if not _is_finite(fitted_value) or fitted_value <= EPS:
            return False
    return True


@njit(fastmath={"contract"}, cache=True)
def _ets_fit(params, data, model):
    alpha, beta, gamma, phi = _extract_ets_params(params, model)
    error_type = model[0]
    trend_type = model[1]
    seasonality_type = model[2]
    seasonal_period = model[3]
    n_timepoints = len(data) - seasonal_period
    level, trend, seasonality = _ets_initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    avg_mean_sq_err_ = 0
    sse_ = 0
    log_fitted_sum = 0.0
    residuals_ = np.zeros(n_timepoints)  # 1 Less residual than data points
    fitted_values_ = np.zeros(n_timepoints)
    if not _ets_params_are_valid(alpha, beta, gamma, phi, trend_type, seasonality_type):
        return (
            LARGE_LOSS,
            level,
            trend,
            seasonality,
            n_timepoints,
            residuals_,
            fitted_values_,
            LARGE_LOSS,
            -LARGE_LOSS,
            0,
        )
    if not _ets_state_is_valid(
        error_type,
        trend_type,
        seasonality_type,
        level,
        trend,
        seasonality[0],
        level,
        0.0,
    ):
        return (
            LARGE_LOSS,
            level,
            trend,
            seasonality,
            n_timepoints,
            residuals_,
            fitted_values_,
            LARGE_LOSS,
            -LARGE_LOSS,
            0,
        )
    for t in range(n_timepoints):
        index = t + seasonal_period
        s_index = t % seasonal_period

        time_point = data[index]

        # Calculate level, trend, and seasonal components
        fitted_value, error, level, trend, seasonality[s_index] = _ets_update_states(
            error_type,
            trend_type,
            seasonality_type,
            level,
            trend,
            seasonality[s_index],
            time_point,
            alpha,
            beta,
            gamma,
            phi,
        )
        if not _ets_state_is_valid(
            error_type,
            trend_type,
            seasonality_type,
            level,
            trend,
            seasonality[s_index],
            fitted_value,
            error,
        ):
            return (
                LARGE_LOSS,
                level,
                trend,
                seasonality,
                n_timepoints,
                residuals_,
                fitted_values_,
                LARGE_LOSS,
                -LARGE_LOSS,
                0,
            )
        residuals_[t] = error
        fitted_values_[t] = fitted_value
        avg_mean_sq_err_ += (time_point - fitted_value) ** 2
        sse_ += error**2
        if error_type == 2:
            log_fitted_sum += np.log(fitted_value)
        if not _is_finite(sse_):
            return (
                LARGE_LOSS,
                level,
                trend,
                seasonality,
                n_timepoints,
                residuals_,
                fitted_values_,
                LARGE_LOSS,
                -LARGE_LOSS,
                0,
            )
    avg_mean_sq_err_ /= n_timepoints
    if not _ets_forecast_cycle_is_valid(
        error_type,
        trend_type,
        seasonality_type,
        level,
        trend,
        seasonality,
        phi,
        n_timepoints,
        seasonal_period,
    ):
        return (
            LARGE_LOSS,
            level,
            trend,
            seasonality,
            n_timepoints,
            residuals_,
            fitted_values_,
            LARGE_LOSS,
            -LARGE_LOSS,
            0,
        )
    variance = sse_ / n_timepoints
    likelihood = -0.5 * n_timepoints * (np.log(2 * np.pi) + np.log(variance) + 1)
    if error_type == 2:
        likelihood -= log_fitted_sum
    # Count phi when it is estimated, even if the estimate is exactly 1.
    k_ = (
        seasonal_period * (seasonality_type != 0)
        + (1 + _ets_is_damped(model)) * (trend_type != 0)
        + 2
    )
    aic = -2 * likelihood + 2 * k_
    return (
        aic,
        level,
        trend,
        seasonality,
        n_timepoints,
        residuals_,
        fitted_values_,
        avg_mean_sq_err_,
        likelihood,
        k_,
    )


@njit(fastmath=True, cache=True)
def _ets_initialise(trend_type, seasonality_type, seasonal_period, data):
    """
    Initialize level, trend, and seasonality values for the ETS model.

    The recursions in ``_ets_fit`` start at observation ``seasonal_period``, so
    the states describe the series at time ``seasonal_period - 1``. The trend is
    the change per time step (additive) or the growth factor per time step
    (multiplicative), following the heuristic in Hyndman et al. [1]_.

    Seasonal models use the first two seasons. The trend is the mean change
    between corresponding points of the two seasons divided by the period, the
    mean of the first season is taken as the level at its centre and moved
    forward to its end, and the seasonal indices are the first season with that
    trend line removed. Non-seasonal models with a trend use a least squares
    line through the first ten observations (on the log scale for a
    multiplicative trend).

    Parameters
    ----------
    data : array-like
        The time series data
        (should contain at least two full seasons if seasonality is specified)

    References
    ----------
    .. [1] R. J. Hyndman, A. B. Koehler, J. K. Ord and R. D. Snyder, Forecasting
       with Exponential Smoothing: The State Space Approach. Springer, 2008.
    """
    m = seasonal_period
    if m > 1:
        first_mean = np.mean(data[:m])
        centre = (m - 1) / 2.0
        if trend_type == 1:
            trend = np.mean(data[m : 2 * m] - data[:m]) / m
        elif trend_type == 2:
            ratio = safe_div(np.mean(data[m : 2 * m]), first_mean)
            trend = ratio ** (1.0 / m) if ratio > 0 else 1.0
        else:
            trend = 0.0
        # Trend line through the first season, centred on its mean
        line = np.empty(m)
        for i in range(m):
            if trend_type == 1:
                line[i] = first_mean + trend * (i - centre)
            elif trend_type == 2:
                line[i] = first_mean * trend ** (i - centre)
            else:
                line[i] = first_mean
        level = line[m - 1]
        if seasonality_type == 1:
            seasonality = data[:m] - line
        elif seasonality_type == 2:
            seasonality = np.empty(m)
            for i in range(m):
                seasonality[i] = safe_div(data[i], line[i])
        else:
            seasonality = np.zeros(m, dtype=np.float64)
        return level, trend, seasonality

    seasonality = np.zeros(1, dtype=np.float64)
    if trend_type == 0:
        return data[0], 0.0, seasonality
    k = min(10, len(data))
    if k < 2:
        return data[0], 0.0 if trend_type == 1 else 1.0, seasonality
    use_log = trend_type == 2
    if use_log:
        for i in range(k):
            if data[i] <= 0.0:
                # A multiplicative trend needs positive data; leave the
                # invalid state for the validity checks in _ets_fit to reject.
                return data[0], 1.0, seasonality
    x_mean = (k - 1) / 2.0
    y_mean = 0.0
    for i in range(k):
        y_mean += np.log(data[i]) if use_log else data[i]
    y_mean /= k
    sxy = 0.0
    sxx = 0.0
    for i in range(k):
        yi = np.log(data[i]) if use_log else data[i]
        sxy += (i - x_mean) * (yi - y_mean)
        sxx += (i - x_mean) ** 2
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if use_log:
        return np.exp(intercept), np.exp(slope), seasonality
    return intercept, slope, seasonality


@njit(fastmath=True, cache=True)
def _ets_update_states(
    error_type,
    trend_type,
    seasonality_type,
    level,
    trend,
    seasonality,
    data_item,
    alpha,
    beta,
    gamma,
    phi,
):
    """
    Update level, trend, and seasonality components.

    Using state space equations for an ETS model.

    Parameters
    ----------
    data_item: float
        The current value of the time series.
    seasonal_index: int
        The index to update the seasonal component.
    """
    # Retrieve the current state values
    curr_level = level
    curr_seasonality = seasonality
    fitted_value, damped_trend, trend_level_combination = _ets_predict_value(
        trend_type, seasonality_type, level, trend, seasonality, phi
    )
    # Calculate the error term (observed value - fitted value)
    if error_type == 2:
        error = safe_div(data_item, fitted_value) - 1  # Multiplicative error
    else:
        error = data_item - fitted_value  # Additive error
    # Update level
    if error_type == 2:
        level = trend_level_combination * (1 + alpha * error)
        trend = damped_trend * (1 + beta * error)
        seasonality = curr_seasonality * (1 + gamma * error)
        if seasonality_type == 1:
            level += alpha * error * curr_seasonality  # Add seasonality correction
            seasonality += gamma * error * trend_level_combination
            if trend_type == 1:
                trend += (curr_level + curr_seasonality) * beta * error
            else:
                trend += safe_div(curr_seasonality, curr_level) * beta * error
        elif trend_type == 1:
            trend += curr_level * beta * error
    else:
        level_correction = 1
        trend_correction = 1
        seasonality_correction = 1
        if seasonality_type == 2:
            # Add seasonality correction
            level_correction *= curr_seasonality
            trend_correction *= curr_seasonality
            seasonality_correction *= trend_level_combination
        if trend_type == 2:
            trend_correction *= curr_level
        level = trend_level_combination + alpha * safe_div(error, level_correction)
        trend = damped_trend + beta * safe_div(error, trend_correction)
        seasonality = curr_seasonality + gamma * safe_div(error, seasonality_correction)
    return (fitted_value, error, level, trend, seasonality)


@njit(fastmath=True, cache=True)
def _ets_predict_value(trend_type, seasonality_type, level, trend, seasonality, phi):
    """

    Generate various useful values, including the next fitted value.

    Parameters
    ----------
    trend : float
        The current trend value for the model
    level : float
        The current level value for the model
    seasonality : float
        The current seasonality value for the model
    phi : float
        The damping parameter for the model

    Returns
    -------
    fitted_value : float
        single prediction based on the current state variables.
    damped_trend : float
        The damping parameter combined with the trend dependent on the model type
    trend_level_combination : float
        Combination of the trend and level based on the model type.
    """
    # Apply damping parameter and
    # calculate commonly used combination of trend and level components
    if trend_type == 2:  # Multiplicative
        damped_trend = trend**phi
        trend_level_combination = level * damped_trend
    else:  # Additive trend, if no trend, then trend = 0
        damped_trend = trend * phi
        trend_level_combination = level + damped_trend

    # Calculate forecast (fitted value) based on the current components
    if seasonality_type == 2:  # Multiplicative
        fitted_value = trend_level_combination * seasonality
    else:  # Additive seasonality, if no seasonality, then seasonality = 0
        fitted_value = trend_level_combination + seasonality
    return fitted_value, damped_trend, trend_level_combination
