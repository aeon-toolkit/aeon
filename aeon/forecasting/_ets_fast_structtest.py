"""ETSForecaster class.

An implementation of the exponential smoothing statistics forecasting algorithm.
Implements additive and multiplicative error models,
None, additive and multiplicative (including damped) trend and
None, additive and mutliplicative seasonality

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = ["ETSForecaster", "ModelType"]

import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from aeon.forecasting.base import BaseForecaster

NONE = 0
ADDITIVE = 1
MULTIPLICATIVE = 2


@jitclass
class ModelType:
    """
    Class describing the error, trend and seasonality model of an ETS forecaster.

    Attributes
    ----------
    error_type : int
        The type of error model; either Additive(1) or Multiplicative(2)
    trend_type : int
        The type of trend model; one of None(0), additive(1) or multiplicative(2).
    seasonality_type : int
        The type of seasonality model; one of None(0), additive(1) or multiplicative(2).
    seasonal_period : int
        The period of the seasonality (m) (e.g., for quaterly data seasonal_period = 4).
    """

    error_type: int
    trend_type: int
    seasonality_type: int
    seasonal_period: int

    def __init__(
        self,
        error_type=ADDITIVE,
        trend_type=NONE,
        seasonality_type=NONE,
        seasonal_period=1,
    ):
        if error_type == NONE:
            raise ValueError("Error must be either additive or multiplicative")
        if seasonal_period < 1 or seasonality_type == NONE:
            seasonal_period = 1
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period


@jitclass([("seasonality", float64[:])])
class StateVariables:
    """
    Class describing the state variables of an ETS forecaster model.

    Attributes
    ----------
    level : float
        The current value of the level (l) state variable
    trend : float
        The current value of the trend (b) state variable
    seasonality : float[]
        The current value of the seasonality (s) state variable
    """

    level: float
    trend: float

    def __init__(self, level=0, trend=0, seasonality=None):
        self.level = level
        self.trend = trend
        if seasonality is None:
            self.seasonality = np.zeros(1, dtype=np.float64)
        else:
            self.seasonality = seasonality


@jitclass
class SmoothingParameters:
    """
    Class describing the smoothing parameters of an ETS forecaster model.

    Attributes
    ----------
    alpha : float, default = 0.1
        Level smoothing parameter.
    beta : float, default = 0.01
        Trend smoothing parameter.
    gamma : float, default = 0.01
        Seasonal smoothing parameter.
    phi : float, default = 0.99
        Trend damping smoothing parameters
    """

    alpha: float
    beta: float
    gamma: float
    phi: float

    def __init__(self, alpha=0.1, beta=0.01, gamma=0.01, phi=0.99):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi


class ETSForecaster(BaseForecaster):
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
    """

    default_model_type = ModelType()
    default_smoothing_parameters = SmoothingParameters()

    def __init__(
        self,
        model_type=default_model_type,
        smoothing_parameters=default_smoothing_parameters,
        horizon=1,
    ):
        self.smoothing_parameters = smoothing_parameters
        if model_type.trend_type == NONE:
            self.smoothing_parameters.beta = 0
        if model_type.seasonality_type == NONE:
            self.smoothing_parameters.gamma = 0
        self.forecast_val_ = 0.0
        self.states = StateVariables()
        self.n_timepoints = 0
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        self.residuals_ = []
        self.model_type = model_type
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
        self.n_timepoints = len(data)
        # print(typeof(self.states.level))
        # print(typeof(data))
        # print(typeof(self.states.seasonality))
        # print(typeof(np.full(self.model_type.seasonal_period, self.states.level)))
        # print(typeof(data[: self.model_type.seasonal_period]))
        _initialise(self.model_type, self.states, data)
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        mul_liklihood_pt2 = 0
        self.residuals_ = np.zeros(
            self.n_timepoints
        )  # 1 Less residual than data points
        for t, data_item in enumerate(data[self.model_type.seasonal_period :]):
            # Calculate level, trend, and seasonal components
            fitted_value, error = _update_states(
                self.model_type,
                self.states,
                data_item,
                t % self.model_type.seasonal_period,
                self.smoothing_parameters,
            )
            self.residuals_[t] = error
            self.avg_mean_sq_err_ += (data_item - fitted_value) ** 2
            self.liklihood_ += error * error
            mul_liklihood_pt2 += np.log(np.fabs(fitted_value))
        self.avg_mean_sq_err_ /= self.n_timepoints - self.model_type.seasonal_period
        self.liklihood_ = (
            self.n_timepoints - self.model_type.seasonal_period
        ) * np.log(self.liklihood_)
        if self.model_type.error_type == MULTIPLICATIVE:
            self.liklihood_ += 2 * mul_liklihood_pt2
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
        # Generate forecasts based on the final values of level, trend, and seasonals
        if self.smoothing_parameters.phi == 1:  # No damping case
            phi_h = float(self.horizon)
        else:
            # Geometric series formula for calculating phi + phi^2 + ... + phi^h
            phi_h = (
                self.smoothing_parameters.phi
                * (1 - self.smoothing_parameters.phi**self.horizon)
                / (1 - self.smoothing_parameters.phi)
            )
        seasonal_index = (
            self.n_timepoints + self.horizon
        ) % self.model_type.seasonal_period
        fitted_value = _predict_value(
            self.model_type, self.states, seasonal_index, phi_h
        )[0]
        return fitted_value


@njit
def _initialise(model: ModelType, states: StateVariables, data):
    """
    Initialize level, trend, and seasonality values for the ETS model.

    Parameters
    ----------
    data : array-like
        The time series data
        (should contain at least two full seasons if seasonality is specified)
    """
    # Initial Level: Mean of the first season
    states.level = np.mean(data[: model.seasonal_period])
    # Initial Trend
    if model.trend_type == ADDITIVE:
        # Average difference between corresponding points in the first two seasons
        states.trend = np.mean(
            data[model.seasonal_period : 2 * model.seasonal_period]
            - data[: model.seasonal_period]
        )
    elif model.trend_type == MULTIPLICATIVE:
        # Average ratio between corresponding points in the first two seasons
        states.trend = np.mean(
            data[model.seasonal_period : 2 * model.seasonal_period]
            / data[: model.seasonal_period]
        )
    else:
        # No trend
        states.trend = 0
    # Initial Seasonality
    if model.seasonality_type == ADDITIVE:
        # Seasonal component is the difference
        # from the initial level for each point in the first season
        states.seasonality = data[: model.seasonal_period] - states.level
    elif model.seasonality_type == MULTIPLICATIVE:
        # Seasonal component is the ratio of each point in the first season
        # to the initial level
        states.seasonality = data[: model.seasonal_period] / states.level
    else:
        # No seasonality
        states.seasonality = np.zeros(1)


@njit
def _update_states(
    model: ModelType,
    states: StateVariables,
    data_item: int,
    seasonal_index: int,
    parameters: SmoothingParameters,
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
    level = states.level
    seasonality = states.seasonality[seasonal_index]
    fitted_value, damped_trend, trend_level_combination = _predict_value(
        model, states, seasonal_index, parameters.phi
    )
    # Calculate the error term (observed value - fitted value)
    if model.error_type == MULTIPLICATIVE:
        error = data_item / fitted_value - 1  # Multiplicative error
    else:
        error = data_item - fitted_value  # Additive error
    # Update level
    if model.error_type == MULTIPLICATIVE:
        states.level = trend_level_combination * (1 + parameters.alpha * error)
        states.trend = damped_trend * (1 + parameters.beta * error)
        states.seasonality[seasonal_index] = seasonality * (
            1 + parameters.gamma * error
        )
        if model.seasonality_type == ADDITIVE:
            states.level += (
                parameters.alpha * error * seasonality
            )  # Add seasonality correction
            states.seasonality[seasonal_index] += (
                parameters.gamma * error * trend_level_combination
            )
            if model.trend_type == ADDITIVE:
                states.trend += (level + seasonality) * parameters.beta * error
            else:
                states.trend += seasonality / level * parameters.beta * error
        elif model.trend_type == ADDITIVE:
            states.trend += level * parameters.beta * error
    else:
        level_correction = 1
        trend_correction = 1
        seasonality_correction = 1
        if model.seasonality_type == MULTIPLICATIVE:
            # Add seasonality correction
            level_correction *= seasonality
            trend_correction *= seasonality
            seasonality_correction *= trend_level_combination
        if model.trend_type == MULTIPLICATIVE:
            trend_correction *= level
        states.level = (
            trend_level_combination + parameters.alpha * error / level_correction
        )
        states.trend = damped_trend + parameters.beta * error / trend_correction
        states.seasonality[seasonal_index] = (
            seasonality + parameters.gamma * error / seasonality_correction
        )
    return (fitted_value, error)


@njit
def _predict_value(
    model: ModelType, states: StateVariables, seasonality_index: int, phi
):
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
    if model.trend_type == MULTIPLICATIVE:
        damped_trend = states.trend**phi
        trend_level_combination = states.level * damped_trend
    else:  # Additive trend, if no trend, then trend = 0
        damped_trend = states.trend * phi
        trend_level_combination = states.level + damped_trend

    # Calculate forecast (fitted value) based on the current components
    if model.seasonality_type == MULTIPLICATIVE:
        fitted_value = trend_level_combination * states.seasonality[seasonality_index]
    else:  # Additive seasonality, if no seasonality, then seasonality = 0
        fitted_value = trend_level_combination + states.seasonality[seasonality_index]
    return fitted_value, damped_trend, trend_level_combination
