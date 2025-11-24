"""Loss functions for optimiser."""

import math

import numpy as np
from numba import njit

from aeon.forecasting.utils._extract_paras import (
    _extract_arma_params,
    _extract_ets_params,
)

LOG_2PI = 1.8378770664093453
EPS = np.float64(1e-8)


@njit(cache=True, fastmath=True)
def _arima_fit(params, data, model):
    """Calculate the AIC of an ARIMA model given the parameters."""
    formatted_params = _extract_arma_params(params, model)  # Extract parameters

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
    sse = 0.0
    start = max(p, q)
    for i in range(start, n):
        sse += residuals[i] * residuals[i]
    variance = sse / (n - start)
    likelihood = (n - start) * (LOG_2PI + np.log(variance) + 1.0)
    k = len(params)
    return likelihood + 2 * k


@njit(fastmath=True, cache=True)
def _ets_fit(params, data, model):
    alpha, beta, gamma, phi = _extract_ets_params(params, model)
    error_type = model[0]
    trend_type = model[1]
    seasonality_type = model[2]
    seasonal_period = model[3]
    n_timepoints = len(data) - seasonal_period
    sum1 = 0.0
    sum2 = 0.0
    for i in range(seasonal_period):
        sum1 += data[i]
        sum2 += data[i + seasonal_period]
    level = sum1 / seasonal_period
    level2 = sum2 / seasonal_period
    # Initial Trend
    if trend_type == 1:
        # Average difference between corresponding points in the first two seasons
        trend = level2 - level
    elif trend_type == 2:
        # Average ratio between corresponding points in the first two seasons
        trend = level2 / level
    else:
        # No trend
        trend = 0
    # Initial Seasonality
    seasonality = np.empty(seasonal_period, dtype=np.float64)
    if seasonality_type == 1:
        # Seasonal component is the difference
        # from the initial level for each point in the first season
        for i in range(seasonal_period):
            seasonality[i] = data[i] - level
    elif seasonality_type == 2:
        # Seasonal component is the ratio of each point in the first season
        # to the initial level
        if level == 0:
            for i in range(seasonal_period):
                seasonality[i] = data[i]
        else:
            for i in range(seasonal_period):
                seasonality[i] = data[i] / level
    else:
        # No seasonality
        seasonality = np.zeros(1, dtype=np.float64)
    avg_mean_sq_err_ = 0
    liklihood_ = 0
    residuals_ = np.zeros(n_timepoints)  # 1 Less residual than data points
    fitted_values_ = np.zeros(n_timepoints)
    s_index = 0
    for t in range(n_timepoints):
        index = t + seasonal_period

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
        s_index += 1
        if s_index == seasonal_period:
            s_index = 0
        residuals_[t] = error
        fitted_values_[t] = fitted_value
        avg_mean_sq_err_ += (time_point - fitted_value) ** 2
        liklihood_error = error
        if error_type == 2:  # Multiplicative
            liklihood_error *= fitted_value
        if liklihood_error > 1e4 or liklihood_ > 1e8:
            liklihood_ = 1e8
            break
        liklihood_ += liklihood_error**2
    avg_mean_sq_err_ /= n_timepoints
    liklihood_ = n_timepoints * math.log(liklihood_)
    k_ = (
        seasonal_period * (seasonality_type != 0)
        + 2 * (trend_type != 0)
        + 2
        + 1 * (phi != 1)
    )
    aic_ = liklihood_ + 2 * k_ - n_timepoints * math.log(n_timepoints)
    return (
        aic_,
        level,
        trend,
        seasonality,
        n_timepoints,
        residuals_,
        fitted_values_,
        avg_mean_sq_err_,
        liklihood_,
        k_,
    )


@njit(inline="always", cache=True)
def safe_div(num, den):
    if den < EPS:
        return num / EPS
    else:
        return num / den


@njit(fastmath=True, cache=True)
def _ets_aic(params, data, model):
    alpha, beta, gamma, phi = _extract_ets_params(params, model)
    error_type = model[0]
    trend_type = model[1]
    seasonality_type = model[2]
    seasonal_period = model[3]
    n_timepoints = len(data) - seasonal_period
    level, trend, seasonality = _ets_initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    liklihood_ = 0
    s_index = 0
    for t in range(n_timepoints):
        index = t + seasonal_period
        # Calculate level, trend, and seasonal components
        # Retrieve the current state values
        curr_level = level
        curr_seasonality = seasonality[s_index]
        if trend_type == 2:  # Multiplicative
            if trend < 0:
                damped_trend = -((-trend) ** phi)
            else:
                damped_trend = trend**phi
            trend_level_combination = level * damped_trend
        else:  # Additive trend, if no trend, then trend = 0
            damped_trend = trend * phi
            trend_level_combination = level + damped_trend

        # Calculate forecast (fitted value) based on the current components
        if seasonality_type == 2:  # Multiplicative
            fitted_value = trend_level_combination * seasonality[s_index]
        else:  # Additive seasonality, if no seasonality, then seasonality = 0
            fitted_value = trend_level_combination + seasonality[s_index]
        # Calculate the error term (observed value - fitted value)
        if error_type == 2:
            error = safe_div(data[index], fitted_value) - 1  # Multiplicative error
        else:
            error = data[index] - fitted_value  # Additive error
        # Update level
        if error_type == 2:
            level = trend_level_combination * (1 + alpha * error)
            trend = damped_trend * (1 + beta * error)
            seasonality[s_index] = curr_seasonality * (1 + gamma * error)
            if seasonality_type == 1:
                level += alpha * error * curr_seasonality  # Add seasonality correction
                seasonality[s_index] += gamma * error * trend_level_combination
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
            seasonality[s_index] = curr_seasonality + gamma * safe_div(
                error, seasonality_correction
            )
        s_index += 1
        if s_index == seasonal_period:
            s_index = 0
        if error_type == 2:  # Multiplicative
            error *= fitted_value
        if error > 1e4 or liklihood_ > 1e8:
            liklihood_ = 1e8
            break
        liklihood_ += error * error
    k_ = (
        seasonal_period
        if (seasonality_type != 0)
        else 0 + 2 if (trend_type != 0) else 0 + 2 + 1 if (phi != 1) else 0
    )
    aic_ = (
        n_timepoints * math.log(liklihood_)
        + 2 * k_
        - n_timepoints * math.log(n_timepoints)
    )
    return aic_


# Fastmath deliberately set to False to avoid issues with numerical stability (ZDEs)
@njit(fastmath=True, cache=True)
def _ets_initialise(trend_type, seasonality_type, seasonal_period, data):
    """
    Initialize level, trend, and seasonality values for the ETS model.

    Parameters
    ----------
    data : array-like
        The time series data
        (should contain at least two full seasons if seasonality is specified)
    """
    # Initial Level: Mean of the first season
    sum1 = 0.0
    sum2 = 0.0
    for i in range(seasonal_period):
        sum1 += data[i]
        sum2 += data[i + seasonal_period]
    level = sum1 / seasonal_period
    level2 = sum2 / seasonal_period
    # Initial Trend
    if trend_type == 1:
        # Average difference between corresponding points in the first two seasons
        trend = level2 - level
    elif trend_type == 2:
        # Average ratio between corresponding points in the first two seasons
        trend = level2 / level
    else:
        # No trend
        trend = 0
    # Initial Seasonality
    seasonality = np.empty(seasonal_period, dtype=np.float64)
    if seasonality_type == 1:
        # Seasonal component is the difference
        # from the initial level for each point in the first season
        for i in range(seasonal_period):
            seasonality[i] = data[i] - level
    elif seasonality_type == 2:
        # Seasonal component is the ratio of each point in the first season
        # to the initial level
        if level == 0:
            for i in range(seasonal_period):
                seasonality[i] = data[i]
        else:
            for i in range(seasonal_period):
                seasonality[i] = data[i] / level
    else:
        # No seasonality
        seasonality = np.zeros(1, dtype=np.float64)
    return level, trend, seasonality


# Fastmath deliberately set to False to avoid issues with numerical stability (ZDEs)
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
    if trend_type == 2:  # Multiplicative
        if trend < 0:
            damped_trend = -((-trend) ** phi)
        else:
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
    # Calculate the error term (observed value - fitted value)
    if error_type == 2:
        if fitted_value < EPS:
            fitted_value = EPS
        error = data_item / fitted_value - 1  # Multiplicative error
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
                trend += curr_seasonality / max(curr_level, EPS) * beta * error
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
        level = trend_level_combination + alpha * error / max(level_correction, EPS)
        trend = damped_trend + beta * error / max(trend_correction, EPS)
        seasonality = curr_seasonality + gamma * error / max(
            seasonality_correction, EPS
        )
    return (fitted_value, error, level, trend, seasonality)


# Fastmath deliberately set to False to avoid issues with numerical stability (ZDEs)
@njit(inline="always", fastmath=True, cache=True)
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
        if trend < 0:
            damped_trend = -((-trend) ** phi)
        else:
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
