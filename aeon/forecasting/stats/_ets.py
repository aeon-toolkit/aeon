"""ETS class.

An implementation of the exponential smoothing statistics forecasting algorithm.
Implements additive and multiplicative error models. We recommend using the AutoETS
version, but this is useful for demonstrations.
"""

__maintainer__ = []
__all__ = ["ETS"]

from typing import Union

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster

ADDITIVE = "additive"
MULTIPLICATIVE = "multiplicative"


class ETS(BaseForecaster):
    """Exponential Smoothing (ETS) forecaster.

    Implements the ETS (Error, Trend, Seasonality) forecaster, supporting additive
    and multiplicative forms of error, trend (including damped), and seasonality
    components. Based on the state space model formulation of exponential
    smoothing as described in Hyndman and Athanasopoulos [1]_.

    Parameters
    ----------
    error_type : string or int, default=1
        Type of error model: 'additive' (0) or 'multiplicative' (1)
    trend_type : string, int or None, default=0
        Type of trend component: None (0), `additive' (1) or 'multiplicative' (2)
    seasonality_type : string or None, default=0
        Type of seasonal component: None (0), `additive' (1) or 'multiplicative' (2)
    seasonal_period : int, default=1
        Number of time points in a seasonal cycle.
    alpha : float, default=0.1
        Level smoothing parameter.
    beta : float, default=0.01
        Trend smoothing parameter.
    gamma : float, default=0.01
        Seasonal smoothing parameter.
    phi : float, default=0.99
        Trend damping parameter (used only for damped trend models).

    Attributes
    ----------
    forecast_val_ : float
        Forecast value for the given horizon.
    level_ : float
        Estimated level component.
    trend_ : float
        Estimated trend component.
    seasonality_ : array-like or None
        Estimated seasonal components.
    aic_ : float
        Akaike Information Criterion of the fitted model.
    avg_mean_sq_err_ : float
        Average mean squared error of the fitted model.
    residuals_ : list of float
        Residuals from the fitted model.
    fitted_values_ : list of float
        Fitted values for the training data.
    liklihood_ : float
        Log-likelihood of the fitted model.
    n_timepoints_ : int
        Number of time points in the training series.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice, 2nd Edition. OTexts, 2014.
       https://otexts.com/fpp3/

    Examples
    --------
    >>> from aeon.forecasting.stats import ETS
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ETS(
    ...     alpha=0.4, beta=0.2, gamma=0.5, phi=0.8,
    ...     error_type='additive', trend_type='multiplicative',
    ...     seasonality_type='multiplicative', seasonal_period=4
    ... )
    >>> forecaster.forecast(y)
    365.5141941111267
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(
        self,
        error_type: Union[int, str] = 1,
        trend_type: Union[int, str, None] = 0,
        seasonality_type: Union[int, str, None] = 0,
        seasonal_period: int = 1,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.01,
        phi: float = 0.99,
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
        super().__init__(horizon=1, axis=1)

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
            Fitted ETS.
        """
        _validate_parameter(self.error_type, False)
        _validate_parameter(self.seasonality_type, True)
        _validate_parameter(self.trend_type, True)

        # Convert to string parameters to ints for numba efficiency
        def _get_int(x):
            if x is None:
                return 0
            if x == ADDITIVE:
                return 1
            if x == MULTIPLICATIVE:
                return 2
            return x

        self._error_type = _get_int(self.error_type)
        self._seasonality_type = _get_int(self.seasonality_type)
        self._trend_type = _get_int(self.trend_type)
        if self._seasonal_period < 1 or self._seasonality_type == 0:
            self._seasonal_period = 1

        if self._trend_type == 0:
            # Required for the equations in _update_states to work correctly
            self._beta = 0
        if self._seasonality_type == 0:
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
        ) = _numba_fit(
            data,
            self._error_type,
            self._trend_type,
            self._seasonality_type,
            self._seasonal_period,
            self.alpha,
            self._beta,
            self._gamma,
            self.phi,
        )
        self.forecast_ = _numba_predict(
            self._trend_type,
            self._seasonality_type,
            self.level_,
            self.trend_,
            self.seasonality_,
            self.phi,
            self.horizon,
            self.n_timepoints_,
            self._seasonal_period,
        )

        return self

    def _predict(self, y, exog=None):
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
        return self.forecast_

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
            self._trend_type, self._seasonality_type, self._seasonal_period, data
        )

    def iterative_forecast(self, y, prediction_horizon):
        """Forecast with ETS specific iterative method.

        Overrides the base class iterative_forecast to avoid refitting on each step.
        This simply rolls the ETS model forward
        """
        self.fit(y)
        preds = np.zeros(prediction_horizon)
        preds[0] = self.forecast_
        for i in range(1, prediction_horizon):
            preds[i] = _numba_predict(
                self._trend_type,
                self._seasonality_type,
                self.level_,
                self.trend_,
                self.seasonality_,
                self.phi,
                i + 1,
                self.n_timepoints_,
                self._seasonal_period,
            )
        return preds


@njit(fastmath=True, cache=True)
def _numba_fit(
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
    n_timepoints = len(data) - seasonal_period
    level, trend, seasonality = _initialise(
        trend_type, seasonality_type, seasonal_period, data
    )
    avg_mean_sq_err_ = 0
    liklihood_ = 0
    residuals_ = np.zeros(n_timepoints)  # 1 Less residual than data points
    fitted_values_ = np.zeros(n_timepoints)
    for t in range(n_timepoints):
        index = t + seasonal_period
        s_index = t % seasonal_period

        time_point = data[index]

        # Calculate level, trend, and seasonal components
        fitted_value, error, level, trend, seasonality[s_index] = _update_states(
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
        residuals_[t] = error
        fitted_values_[t] = fitted_value
        avg_mean_sq_err_ += (time_point - fitted_value) ** 2
        liklihood_error = error
        if error_type == 2:  # Multiplicative
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


@njit(fastmath=True, cache=True)
def _numba_predict(
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
        phi_h = horizon
    else:
        # Geometric series formula for calculating phi + phi^2 + ... + phi^h
        phi_h = phi * (1 - phi**horizon) / (1 - phi)
    seasonal_index = (n_timepoints + horizon - 1) % seasonal_period
    return _predict_value(
        trend_type,
        seasonality_type,
        level,
        trend,
        seasonality[seasonal_index],
        phi_h,
    )[0]


@njit(fastmath=True, cache=True)
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
    if trend_type == 1:
        # Average difference between corresponding points in the first two seasons
        trend = np.mean(
            data[seasonal_period : 2 * seasonal_period] - data[:seasonal_period]
        )
    elif trend_type == 2:
        # Average ratio between corresponding points in the first two seasons
        trend = np.mean(
            data[seasonal_period : 2 * seasonal_period] / data[:seasonal_period]
        )
    else:
        # No trend
        trend = 0
    # Initial Seasonality
    if seasonality_type == 1:
        # Seasonal component is the difference
        # from the initial level for each point in the first season
        seasonality = data[:seasonal_period] - level
    elif seasonality_type == 2:
        # Seasonal component is the ratio of each point in the first season
        # to the initial level
        seasonality = data[:seasonal_period] / level
    else:
        # No seasonality
        seasonality = np.zeros(1, dtype=np.float64)
    return level, trend, seasonality


@njit(fastmath=True, cache=True)
def _update_states(
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
    fitted_value, damped_trend, trend_level_combination = _predict_value(
        trend_type, seasonality_type, level, trend, seasonality, phi
    )
    # Calculate the error term (observed value - fitted value)
    if error_type == 2:
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
                trend += curr_seasonality / curr_level * beta * error
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
        level = trend_level_combination + alpha * error / level_correction
        trend = damped_trend + beta * error / trend_correction
        seasonality = curr_seasonality + gamma * error / seasonality_correction
    return (fitted_value, error, level, trend, seasonality)


@njit(fastmath=True, cache=True)
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


def _validate_parameter(var, can_be_none):
    valid_str = (ADDITIVE, MULTIPLICATIVE)
    valid_int = (1, 2)
    if can_be_none:
        valid_str = (None, ADDITIVE, MULTIPLICATIVE)
        valid_int = (0, 1, 2)
    valid = True
    if isinstance(var, str) or var is None:
        if var not in valid_str:
            valid = False
    elif isinstance(var, int):
        if var not in valid_int:
            valid = False
    else:
        valid = False
    if not valid:
        raise ValueError(
            f"variable must be either string or integer with values"
            f" {valid_str} or {valid_int} but saw {var}"
        )
