"""ETSForecaster class.

An implementation of the exponential smoothing statistics forecasting algorithm.
Implements additive and multiplicative error models,
None, additive and multiplicative (including damped) trend and
None, additive and mutliplicative seasonality

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = ["ETSForecaster"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster

NOGIL = False
CACHE = True

NONE = 0
ADDITIVE = 1
MULTIPLICATIVE = 2


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
    error_type : int
        The type of error model; either Additive(1) or Multiplicative(2)
    trend_type : int
        The type of trend model; one of None(0), additive(1) or multiplicative(2).
    seasonality_type : int
        The type of seasonality model; one of None(0), additive(1) or multiplicative(2).
    seasonal_period : int
        The period of the seasonality (m) (e.g., for quaterly data seasonal_period = 4).

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.

    Examples
    --------
    >>> from aeon.forecasting import ETSForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8, horizon=1,
        error_type=1, trend_type=2, seasonality_type=2, seasonal_period=4)
    >>> forecaster.fit(y)
    ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8, seasonal_period=4,
                  seasonality_type=2, trend_type=2)
    >>> forecaster.predict()
    366.90200486015596
    """

    def __init__(
        self,
        error_type=ADDITIVE,
        trend_type=NONE,
        seasonality_type=NONE,
        seasonal_period=1,
        alpha=0.1,
        beta=0.01,
        gamma=0.01,
        phi=0.99,
        horizon=1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.forecast_val_ = 0.0
        self.level_ = 0.0
        self.trend_ = 0.0
        self.seasonality_ = None
        self._beta = beta
        self._gamma = gamma
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period
        self._seasonal_period = seasonal_period
        self.n_timepoints_ = 0
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        self.k_ = 0
        self.aic_ = 0
        self.residuals_ = []
        self.fitted_values_ = []
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
            Fitted ETSForecaster.
        """
        assert (
            self.error_type != NONE
        ), "Error must be either additive or multiplicative"
        if self._seasonal_period < 1 or self.seasonality_type == NONE:
            self._seasonal_period = 1

        if self.trend_type == NONE:
            # Required for the equations in _update_states to work correctly
            self._beta = 0
        if self.seasonality_type == NONE:
            # Required for the equations in _update_states to work correctly
            self._gamma = 0
        data = y.squeeze()
        (
            self.level_,
            self.trend_,
            self.seasonality_,
            self.n_timepoints_,
            self.residuals_,
            self.fitted_values_,
            self.avg_mean_sq_err_,
            self.liklihood_,
            self.k_,
            self.aic_,
        ) = _fit(
            data,
            self.error_type,
            self.trend_type,
            self.seasonality_type,
            self._seasonal_period,
            self.alpha,
            self._beta,
            self._gamma,
            self.phi,
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
        fitted_value = _predict(
            self.trend_type,
            self.seasonality_type,
            self.level_,
            self.trend_,
            self.seasonality_,
            self.phi,
            self.horizon,
            self.n_timepoints_,
            self._seasonal_period,
        )
        if y is None:
            return np.array([fitted_value])
        else:
            return np.insert(y, 0, fitted_value)[:-1]

    def _initialise(self, data):
        """
        Initialize level, trend, and seasonality values for the ETS model.

        Parameters
        ----------
        data : array-like
            The time series data
            (should contain at least two full seasons if seasonality is specified)
        """
        self.level_, self.trend_, self.seasonality_ = _initialise(
            self.trend_type, self.seasonality_type, self._seasonal_period, data
        )


@njit(nogil=NOGIL, cache=CACHE)
def _fit(
    data,
    error_type,
    trend_type,
    seasonality_type,
    seasonal_period,
    alpha,
    beta,
    gamma,
    phi,
):
    assert error_type != NONE, "Error must be either additive or multiplicative"
    assert (
        error_type != MULTIPLICATIVE or data.min() > 0
    ), "Data must be positive with multiplicative errors"
    if seasonal_period < 1 or seasonality_type == NONE:
        seasonal_period = 1
    if trend_type == NONE:
        # Required for the equations in _update_states to work correctly
        beta = 0
    if seasonality_type == NONE:
        # Required for the equations in _update_states to work correctly
        gamma = 0
    n_timepoints = len(data) - seasonal_period
    level, trend, seasonality = _initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    avg_mean_sq_err_ = 0
    liklihood_ = 0
    residuals_ = np.zeros(n_timepoints)  # 1 Less residual than data points
    fitted_values_ = np.zeros(n_timepoints)
    for t, data_item in enumerate(data[seasonal_period:]):
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
        residuals_[t] = error
        fitted_values_[t] = fitted_value
        avg_mean_sq_err_ += (data_item - fitted_value) ** 2
        liklihood_error = error
        if error_type == MULTIPLICATIVE:
            liklihood_error *= fitted_value
        liklihood_ += liklihood_error**2
    avg_mean_sq_err_ /= n_timepoints
    liklihood_ = n_timepoints * np.log(liklihood_)
    k_ = (
        seasonal_period * (seasonality_type != 0)
        + 2 * (trend_type != 0)
        + 2
        + 1 * (phi != 1)
    )
    aic_ = liklihood_ + 2 * k_ - n_timepoints * np.log(n_timepoints)
    return (
        level,
        trend,
        seasonality,
        n_timepoints,
        residuals_,
        fitted_values_,
        avg_mean_sq_err_,
        liklihood_,
        k_,
        aic_,
    )


@njit(nogil=NOGIL, cache=CACHE)
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
        phi_h = 1
    else:
        # Geometric series formula for calculating phi + phi^2 + ... + phi^h
        phi_h = phi * (1 - phi**horizon) / (1 - phi)
    seasonal_index = (n_timepoints + horizon) % seasonal_period
    return _predict_value(
        trend_type,
        seasonality_type,
        level,
        trend,
        seasonality[seasonal_index],
        phi_h,
    )[0]


@njit(nogil=NOGIL, cache=CACHE)
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
    level = np.mean(data[:seasonal_period])
    # Initial Trend
    if trend_type == ADDITIVE:
        # Average difference between corresponding points in the first two seasons
        trend = np.mean(
            data[seasonal_period : 2 * seasonal_period] - data[:seasonal_period]
        )
    elif trend_type == MULTIPLICATIVE:
        # Average ratio between corresponding points in the first two seasons
        trend = np.mean(
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
        seasonality = np.zeros(1, dtype=np.float64)
    return level, trend, seasonality


@njit(nogil=NOGIL, cache=CACHE)
def _update_states(
    error_type,
    trend_type,
    seasonality_type,
    level,
    trend,
    seasonality,
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
    curr_level = level
    curr_seasonality = seasonality
    fitted_value, damped_trend, trend_level_combination = _predict_value(
        trend_type, seasonality_type, level, trend, seasonality, phi
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
            level_correction *= curr_seasonality
            trend_correction *= curr_seasonality
            seasonality_correction *= trend_level_combination
        if trend_type == MULTIPLICATIVE:
            trend_correction *= curr_level
        level = trend_level_combination + alpha * error / level_correction
        trend = damped_trend + beta * error / trend_correction
        seasonality = curr_seasonality + gamma * error / seasonality_correction
    return (fitted_value, error, level, trend, seasonality)


@njit(nogil=NOGIL, cache=CACHE)
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
        damped_trend = trend**phi
        trend_level_combination = level * damped_trend
    else:  # Additive trend, if no trend, then trend = 0
        damped_trend = trend * phi
        trend_level_combination = level + damped_trend

    # Calculate forecast (fitted value) based on the current components
    if seasonality_type == MULTIPLICATIVE:
        fitted_value = trend_level_combination * seasonality
    else:  # Additive seasonality, if no seasonality, then seasonality = 0
        fitted_value = trend_level_combination + seasonality
    return fitted_value, damped_trend, trend_level_combination
