"""ETSForecaster class.

An implementation of the exponential smoothing statistics forecasting algorithm.
Implements additive and multiplicative error models,
None, additive and multiplicative (including damped) trend and
None, additive and multiplicative seasonality
"""

__maintainer__ = []
__all__ = ["ETSForecaster", "NONE", "ADDITIVE", "MULTIPLICATIVE"]

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

    An implementation of the exponential smoothing forecasting algorithm.
    Implements additive and multiplicative error models, None, additive and
    multiplicative (including damped) trend and None, additive and mutliplicative
    seasonality. See [1]_ for a description.

    Parameters
    ----------
    error_type : int, default = 1
        Either NONE (0), ADDITIVE (1) or MULTIPLICATIVE (2).
    trend_type : int, default = 0
        Either NONE (0), ADDITIVE (1) or MULTIPLICATIVE (2).
    seasonality_type : int, default = 0
        Either NONE (0), ADDITIVE (1) or MULTIPLICATIVE (2).
    seasonal_period : int, default=1
        Length of seasonality period. If seasonality_type is NONE, this is assumed to
        be 1
    alpha : float, default = 0.1
        Level smoothing parameter.
    beta : float, default = 0.01
        Trend smoothing parameter. If trend_type is NONE, this is assumed to be 0.0.
    gamma : float, default = 0.01
        Seasonal smoothing parameter. If seasonality is NONE, this is assumed to be
        0.0.
    phi : float, default = 0.99
        Trend damping smoothing parameters
    horizon : int, default = 1
        The horizon to forecast to.

    Attributes
    ----------
    mean_sq_err_ : float
        Mean squared error.
    likelihood_ : float
        Likelihood of the fitted model based on residuals.
    residuals_ : arraylike
        List of train set differences between fitted and actual values.
    n_timpoints_ : int
        Length of the series passed to fit.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.

    Examples
    --------
    >>> from aeon.forecasting import ETSForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8, horizon=1)
    >>> forecaster.fit(y)
    ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8)
    >>> forecaster.predict()
    449.9435566831507
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
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.mean_sq_err_ = 0
        self.likelihood_ = 0
        self.residuals_ = []
        self.n_timpoints_ = 0
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
        self.n_timepoints_ = len(y)
        if self.error_type != MULTIPLICATIVE and self.error_type != ADDITIVE:
            raise ValueError("Error must be either additive or multiplicative")
        self._seasonal_period = self.seasonal_period
        if self.seasonal_period < 1 or self.seasonality_type == NONE:
            self._seasonal_period = 1
        self._beta = self.beta
        if self.trend_type == NONE or self.trend_type is None:
            self._beta = 0
        self._gamma = self.gamma
        if self.seasonality_type == NONE or self.trend_type is None:
            self._gamma = 0
        data = np.array(y.squeeze(), dtype=np.float64)
        (
            self._level,
            self._trend,
            self._seasonality,
            self.residuals_,
            self.mean_sq_err_,
            self.likelihood_,
        ) = _fit_numba(
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
        exog : np.ndarray, default = None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        return _predict_numba(
            self.trend_type,
            self.seasonality_type,
            self._level,
            self._trend,
            self._seasonality,
            self.phi,
            self.horizon,
            self.n_timepoints_,
            self.seasonal_period,
        )


@njit(nogil=NOGIL, cache=CACHE)
def _fit_numba(
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
    n_timepoints = len(data)
    level, trend, seasonality = _initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    mse = 0
    lhood = 0
    mul_likelihood_pt2 = 0
    res = np.zeros(n_timepoints)  # 1 Less residual than data points
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
        res[t] = error
        mse += (data_item - fitted_value) ** 2
        lhood += error * error
        mul_likelihood_pt2 += np.log(np.fabs(fitted_value))
    mse /= n_timepoints - seasonal_period
    lhood = (n_timepoints - seasonal_period) * np.log(lhood)
    if error_type == MULTIPLICATIVE:
        lhood += 2 * mul_likelihood_pt2
    return level, trend, seasonality, res, mse, lhood


def _predict_numba(
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
        seasonality = np.zeros(1)
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
