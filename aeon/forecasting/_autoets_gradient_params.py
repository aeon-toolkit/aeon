"""AutoETSForecaster class.

Extends the ETSForecaster to automatically calculate the smoothing parameters

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = ["AutoETSForecaster"]

import numpy as np
import torch

from aeon.forecasting._ets_fast import ADDITIVE, MULTIPLICATIVE, NONE
from aeon.forecasting.base import BaseForecaster


class AutoETSForecaster(BaseForecaster):
    """Exponential Smoothing forecaster.

    An implementation of the exponential smoothing statistics forecasting algorithm.
    Implements additive and multiplicative error models,
    None, additive and multiplicative (including damped) trend and
    None, additive and mutliplicative seasonality[1]_.

    Parameters
    ----------
    alpha : float, default = 0.1
        Level smoothing parameter.
    beta : float, default = 0.01
        Trend smoothing parameter.
    gamma : float, default = 0.01
        Seasonal smoothing parameter.
    phi : float, default = 0.99
        Trend damping smoothing parameters
    horizon : int, default = 1
        The horizon to forecast to.
    model_type : ModelType, default = ModelType()
        A object of type ModelType, describing the error,
        trend and seasonality type of this ETS model.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.

    Examples
    --------
    >>> from aeon.forecasting import ETSForecaster, ModelType
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8, horizon=1,
                               model_type=ModelType(1,2,2,4))
    >>> forecaster.fit(y)
    >>> forecaster.predict()
    366.90200486015596
    """

    def __init__(
        self,
        error_type=ADDITIVE,
        trend_type=NONE,
        seasonality_type=NONE,
        seasonal_period=1,
        horizon=1,
    ):
        assert error_type != NONE, "Error must be either additive or multiplicative"
        if seasonal_period < 1 or seasonality_type == NONE:
            seasonal_period = 1
        self.alpha = torch.tensor(0.1, requires_grad=True)  # Level smoothing
        self.beta = torch.tensor(0.05, requires_grad=True)  # Trend smoothing
        self.gamma = torch.tensor(0.05, requires_grad=True)  # Seasonality smoothing
        self.phi = torch.tensor(0.98, requires_grad=True)  # Damping factor
        if trend_type == NONE:
            self.beta = 0
        if seasonality_type == NONE:
            self.gamma = 0
        self.forecast_val_ = 0.0
        self.level = (0,)
        self.trend = (0,)
        self.seasonality = np.zeros(1, dtype=np.float64)
        self.n_timepoints = 0
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        self.residuals_ = []
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit Exponential Smoothing forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        self
            Fitted BaseForecaster.
        """
        data = np.array(y.squeeze(), dtype=np.float64)
        (
            self.level,
            self.trend,
            self.seasonality,
            self.residuals_,
            self.avg_mean_sq_err_,
            self.liklihood_,
        ) = _fit(
            data,
            self.error_type,
            self.trend_type,
            self.seasonality_type,
            self.seasonal_period,
        )
        return self

    def _predict(self, y=None, exog=None):
        """
        Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        y = np.array(y, dtype=np.float64)

        return _predict(
            self.trend_type,
            self.seasonality_type,
            self.level,
            self.trend,
            self.seasonality,
            self.phi,
            self.horizon,
            self.n_timepoints,
            self.seasonal_period,
        )


def _fit(data, error_type, trend_type, seasonality_type, seasonal_period):
    torch.autograd.set_detect_anomaly(True)
    data = torch.tensor(data)
    n_timepoints = len(data)
    # print(typeof(self.states.level))
    # print(typeof(data))
    # print(typeof(self.states.seasonality))
    # print(typeof(np.full(self.model_type.seasonal_period, self.states.level)))
    # print(typeof(data[: self.model_type.seasonal_period]))
    level, trend, seasonality = _initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    alpha = torch.tensor(0.1, requires_grad=True)  # Level smoothing
    beta = torch.tensor(0.05, requires_grad=True)  # Trend smoothing
    gamma = torch.tensor(0.05, requires_grad=True)  # Seasonality smoothing
    phi = torch.tensor(0.98, requires_grad=True)  # Damping factor
    batch_size = seasonal_period * 2
    num_batches = len(data) // batch_size
    # residuals_ = torch.zeros(n_timepoints)  # 1 Less residual than data points
    optimizer = torch.optim.SGD([alpha, beta, gamma, phi], lr=0.001)
    for _epoch in range(100):  # number of epochs
        for i in range(1, num_batches):
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
                # residuals_[t] = error
                liklihood_ += error * error
                mul_liklihood_pt2 += torch.log(torch.abs(fitted_value))
            liklihood_ = (n_timepoints - seasonal_period) * torch.log(liklihood_)
            if error_type == MULTIPLICATIVE:
                liklihood_ += 2 * mul_liklihood_pt2
            liklihood_.backward()
            optimizer.step()
            optimizer.zero_grad()
            level = level.clone().detach()
            trend = trend.clone().detach()
            seasonality = seasonality.clone().detach()
    return alpha, beta, gamma, phi


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
        trend = 0
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
        seasonality = np.zeros(1)
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
        level = trend_level_combination * (1 + alpha * error)
        trend = damped_trend * (1 + beta * error)
        seasonality = curr_seasonality * (1 + gamma * error)
        if seasonality_type == ADDITIVE:
            level += alpha * error * curr_seasonality  # Add seasonality correction
            seasonality += gamma * error * trend_level_combination
            if trend_type == ADDITIVE:
                trend += (curr_level + curr_seasonality) * beta * error
            else:
                trend += curr_seasonality / curr_level * beta * error
        elif trend_type == ADDITIVE:
            trend += curr_level * beta * error
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
