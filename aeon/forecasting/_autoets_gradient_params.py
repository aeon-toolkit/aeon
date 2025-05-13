"""AutoETSForecaster class.

Extends the ETSForecaster to automatically calculate the smoothing parameters

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = []

import torch

from aeon.forecasting._ets_fast import ADDITIVE, MULTIPLICATIVE, NONE, ETSForecaster


def _calc_model_liklihood(
    data, error_type, trend_type, seasonality_type, seasonal_period
):
    alpha, beta, gamma, phi = _optimise_parameters(
        data, error_type, trend_type, seasonality_type, seasonal_period
    )
    forecaster = ETSForecaster(
        error_type,
        trend_type,
        seasonality_type,
        seasonal_period,
        alpha,
        beta,
        gamma,
        phi,
        1,
    )
    forecaster.fit(data)
    return alpha, beta, gamma, phi, forecaster.residuals_, forecaster.liklihood_


def _optimise_parameters(
    data, error_type, trend_type, seasonality_type, seasonal_period
):
    torch.autograd.set_detect_anomaly(True)
    data = torch.tensor(data)
    n_timepoints = len(data)
    if seasonality_type == 0:
        seasonal_period = 1
    level, trend, seasonality = _initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    alpha = torch.tensor(0.1, requires_grad=True)  # Level smoothing
    parameters = [alpha]
    if trend_type == NONE:
        beta = torch.tensor(0)  # Trend smoothing
    else:
        beta = torch.tensor(0.05, requires_grad=True)  # Trend smoothing
        parameters.append(beta)
    if seasonality_type == NONE:
        gamma = torch.tensor(0)  # Trend smoothing
    else:
        gamma = torch.tensor(0.05, requires_grad=True)  # Seasonality smoothing
        parameters.append(gamma)
    phi = torch.tensor(0.98, requires_grad=True)  # Damping factor
    batch_size = len(data)  # seasonal_period * 2
    num_batches = len(data) // batch_size
    # residuals_ = torch.zeros(n_timepoints)  # 1 Less residual than data points
    optimizer = torch.optim.SGD([alpha, beta, gamma, phi], lr=0.01)
    for _epoch in range(10):  # number of epochs
        for i in range(0, num_batches):
            batch_of_data = data[i * batch_size : (i + 1) * batch_size]
            liklihood_ = torch.tensor(0, dtype=torch.float64)
            mul_liklihood_pt2 = torch.tensor(0, dtype=torch.float64)
            for t, data_item in enumerate(batch_of_data):
                # Calculate level, trend, and seasonal components
                fitted_value, error, level, trend, seasonality[t % seasonal_period] = (
                    _update_states(
                        error_type,
                        trend_type,
                        seasonality_type,
                        level,
                        trend,
                        seasonality[t % seasonal_period],
                        data_item,
                        alpha,
                        beta,
                        gamma,
                        phi,
                    )
                )
                liklihood_ += error * error
                mul_liklihood_pt2 += torch.log(torch.abs(fitted_value))
            liklihood_ = (n_timepoints - seasonal_period) * torch.log(liklihood_)
            if error_type == MULTIPLICATIVE:
                liklihood_ += 2 * mul_liklihood_pt2
            liklihood_.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Impose sensible parameter limits
            alpha = alpha.clone().detach().requires_grad_().clamp(0, 1)
            if trend_type != NONE:
                # Impose sensible parameter limits
                beta = beta.clone().detach().requires_grad_().clamp(0, 1)
            if seasonality_type != NONE:
                # Impose sensible parameter limits
                gamma = gamma.clone().detach().requires_grad_().clamp(0, 1)
            # Impose sensible parameter limits
            phi = phi.clone().detach().requires_grad_().clamp(0.1, 0.98)
            level = level.clone().detach()
            trend = trend.clone().detach()
            seasonality = seasonality.clone().detach()
    return alpha.item(), beta.item(), gamma.item(), phi.item()


def _predict(
    trend_type,
    seasonality_type,
    level,
    trend,
    seasonality,
    phi,
    horizon,
    n_timepoints,
    seasonal_period,
):
    # Generate forecasts based on the final values of level, trend, and seasonals
    if phi == 1:  # No damping case
        phi_h = float(horizon)
    else:
        # Geometric series formula for calculating phi + phi^2 + ... + phi^h
        phi_h = phi * (1 - phi**horizon) / (1 - phi)
    seasonal_index = (n_timepoints + horizon) % seasonal_period
    return _predict_value(
        trend_type, seasonality_type, level, trend, seasonality[seasonal_index], phi_h
    )[0]


def _initialise(trend_type, seasonality_type, seasonal_period, data):
    """
    Initialize level, trend, and seasonality values for the ETS model.

    Parameters
    ----------
    data : array-like
        The time series data
        (should contain at least two full seasons if seasonality is specified)
    """
    # Initial Level: Mean of the first season
    level = torch.mean(data[:seasonal_period])
    # Initial Trend
    if trend_type == ADDITIVE:
        # Average difference between corresponding points in the first two seasons
        trend = torch.mean(
            data[seasonal_period : 2 * seasonal_period] - data[:seasonal_period]
        )
    elif trend_type == MULTIPLICATIVE:
        # Average ratio between corresponding points in the first two seasons
        trend = torch.mean(
            data[seasonal_period : 2 * seasonal_period] / data[:seasonal_period]
        )
    else:
        # No trend
        trend = torch.tensor(0)
    # Initial Seasonality
    if seasonality_type == ADDITIVE:
        # Seasonal component is the difference
        # from the initial level for each point in the first season
        seasonality = data[:seasonal_period] - level
    elif seasonality_type == MULTIPLICATIVE:
        # Seasonal component is the ratio of each point in the first season
        # to the initial level
        seasonality = data[:seasonal_period] / level
    else:
        # No seasonality
        seasonality = torch.zeros(1)
    return level, trend, seasonality


def _update_states(
    error_type,
    trend_type,
    seasonality_type,
    curr_level,
    curr_trend,
    curr_seasonality,
    data_item: int,
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
    fitted_value, damped_trend, trend_level_combination = _predict_value(
        trend_type, seasonality_type, curr_level, curr_trend, curr_seasonality, phi
    )
    # Calculate the error term (observed value - fitted value)
    if error_type == MULTIPLICATIVE:
        error = data_item / fitted_value - 1  # Multiplicative error
    else:
        error = data_item - fitted_value  # Additive error
    # Update level
    if error_type == MULTIPLICATIVE:
        level = trend_level_combination.clone() * (1 + alpha.clone() * error.clone())
        trend = damped_trend.clone() * (1 + beta.clone() * error.clone())
        seasonality = curr_seasonality.clone() * (1 + gamma.clone() * error.clone())
        if seasonality_type == ADDITIVE:
            # Add seasonality correction
            level += alpha.clone() * error.clone() * curr_seasonality.clone()
            seasonality += (
                gamma.clone() * error.clone() * trend_level_combination.clone()
            )
            if trend_type == ADDITIVE:
                trend += (
                    (curr_level.clone() + curr_seasonality.clone())
                    * beta.clone()
                    * error.clone()
                )
            else:
                trend += (
                    (curr_seasonality.clone() / curr_level.clone())
                    * beta.clone()
                    * error.clone()
                )
        elif trend_type == ADDITIVE:
            trend += curr_level.clone() * beta.clone() * error.clone()
    else:
        level_correction = 1
        trend_correction = 1
        seasonality_correction = 1
        if seasonality_type == MULTIPLICATIVE:
            # Add seasonality correction
            level_correction *= curr_seasonality.clone()
            trend_correction *= curr_seasonality.clone()
            seasonality_correction *= trend_level_combination.clone()
        if trend_type == MULTIPLICATIVE:
            trend_correction *= curr_level.clone()
        level = (
            trend_level_combination.clone()
            + alpha.clone() * error.clone() / level_correction
        )
        trend = damped_trend.clone() + beta.clone() * error.clone() / trend_correction
        seasonality = (
            curr_seasonality.clone()
            + gamma.clone() * error.clone() / seasonality_correction
        )
    return (fitted_value, error, level, trend, seasonality)


def _predict_value(trend_type, seasonality_type, level, trend, seasonality, phi):
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
        The damping parameter combined with the trend dependant on the model type
    trend_level_combination : float
        Combination of the trend and level based on the model type.
    """
    # Apply damping parameter and
    # calculate commonly used combination of trend and level components
    if trend_type == MULTIPLICATIVE:
        damped_trend = trend.clone() ** phi.clone()
        trend_level_combination = level.clone() * damped_trend.clone()
    else:  # Additive trend, if no trend, then trend = 0
        damped_trend = trend.clone() * phi.clone()
        trend_level_combination = level.clone() + damped_trend.clone()
    # Calculate forecast (fitted value) based on the current components
    if seasonality_type == MULTIPLICATIVE:
        fitted_value = trend_level_combination.clone() * seasonality.clone()
    else:  # Additive seasonality, if no seasonality, then seasonality = 0
        fitted_value = trend_level_combination.clone() + seasonality.clone()
    return fitted_value, damped_trend, trend_level_combination
