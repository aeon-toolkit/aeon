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

from aeon.forecasting.base import BaseForecaster

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
        error_type: int = ADDITIVE,
        trend_type: int = NONE,
        seasonality_type: int = NONE,
        seasonal_period: int = 1,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.01,
        phi: float = 0.99,
        horizon: int = 1,
        error_type: int = ADDITIVE,
        trend_type: int = NONE,
        seasonality_type: int = NONE,
        seasonal_period: int = 1,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.01,
        phi: float = 0.99,
        horizon: int = 1,
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
        self.n_timepoints = 0
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        self.k_ = 0
        self.aic_ = 0
        self.residuals_ = []
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
        assert (
            self.error_type != NONE
        ), "Error must be either additive or multiplicative"
        if self._seasonal_period < 1 or self.seasonality_type == NONE:
            self._seasonal_period = 1
        if self.trend_type == NONE:
            self._beta = (
                0  # Required for the equations in _update_states to work correctly
            )
        if self.seasonality_type == NONE:
            self._gamma = (
                0  # Required for the equations in _update_states to work correctly
            )
        data = y.squeeze()
        self.n_timepoints = len(data)
        self._initialise(data)
        num_vals = self.n_timepoints - self._seasonal_period
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        # 1 Less residual than data points
        self.residuals_ = np.zeros(num_vals)
        for t, data_item in enumerate(data[self._seasonal_period :]):
            # Calculate level, trend, and seasonal components
            fitted_value, error = self._update_states(
                data_item, t % self._seasonal_period
            )
            self.residuals_[t] = error
            self.avg_mean_sq_err_ += (data_item - fitted_value) ** 2
            liklihood_error = error
            if self.error_type == MULTIPLICATIVE:
                liklihood_error *= fitted_value
            self.liklihood_ += liklihood_error**2
        self.avg_mean_sq_err_ /= num_vals
        self.liklihood_ = num_vals * np.log(self.liklihood_)
        self.k_ = (
            self.seasonal_period * (self.seasonality_type != 0)
            + 2 * (self.trend_type != 0)
            + 2
            + 1 * (self.phi != 1)
        )
        self.aic_ = self.liklihood_ + 2 * self.k_ - num_vals * np.log(num_vals)
        return self

    def _update_states(self, data_item, seasonal_index):
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
        level = self.level_
        trend = self.trend_
        seasonality = self.seasonality_[seasonal_index]
        fitted_value, damped_trend, trend_level_combination = self._predict_value(
            level, trend, seasonality, self.phi
        )
        # Calculate the error term (observed value - fitted value)
        if self.error_type == MULTIPLICATIVE:
            error = data_item / fitted_value - 1  # Multiplicative error
        else:
            error = data_item - fitted_value  # Additive error
        # Update level
        if self.error_type == MULTIPLICATIVE:
            self.level_ = trend_level_combination * (1 + self.alpha * error)
            self.trend_ = damped_trend * (1 + self._beta * error)
            self.seasonality_[seasonal_index] = seasonality * (1 + self._gamma * error)
            if self.seasonality_type == ADDITIVE:
                self.level_ += (
                    self.alpha * error * seasonality
                )  # Add seasonality correction
                self.seasonality_[seasonal_index] += (
                    self._gamma * error * trend_level_combination
                )
                if self.trend_type == ADDITIVE:
                    self.trend_ += (level + seasonality) * self._beta * error
                else:
                    self.trend_ += seasonality / level * self._beta * error
            elif self.trend_type == ADDITIVE:
                self.trend_ += level * self._beta * error
        else:
            level_correction = 1
            trend_correction = 1
            seasonality_correction = 1
            if self.seasonality_type == MULTIPLICATIVE:
                # Add seasonality correction
                level_correction *= seasonality
                trend_correction *= seasonality
                seasonality_correction *= trend_level_combination
            if self.trend_type == MULTIPLICATIVE:
                trend_correction *= level
            self.level_ = (
                trend_level_combination + self.alpha * error / level_correction
            )
            self.trend_ = damped_trend + self._beta * error / trend_correction
            self.seasonality_[seasonal_index] = (
                seasonality + self._gamma * error / seasonality_correction
            )
        return (fitted_value, error)

    def _initialise(self, data):
        """
        Initialize level, trend, and seasonality values for the ETS model.

        Parameters
        ----------
        data : array-like
            The time series data
            (should contain at least two full seasons if seasonality is specified)
        """
        # Initial Level: Mean of the first season
        self.level_ = np.mean(data[: self._seasonal_period])
        # Initial Trend
        if self.trend_type == ADDITIVE:
            # Average difference between corresponding points in the first two seasons
            self.trend_ = np.mean(
                data[self._seasonal_period : 2 * self._seasonal_period]
                - data[: self._seasonal_period]
            )
        elif self.trend_type == MULTIPLICATIVE:
            # Average ratio between corresponding points in the first two seasons
            self.trend_ = np.mean(
                data[self._seasonal_period : 2 * self._seasonal_period]
                / data[: self._seasonal_period]
            )
        else:
            # No trend
            self.trend_ = 0
        # Initial Seasonality
        if self.seasonality_type == ADDITIVE:
            # Seasonal component is the difference
            # from the initial level for each point in the first season
            self.seasonality_ = data[: self._seasonal_period] - self.level_
        elif self.seasonality_type == MULTIPLICATIVE:
            # Seasonal component is the ratio of each point in the first season
            # to the initial level
            self.seasonality_ = data[: self._seasonal_period] / self.level_
        else:
            # No seasonality
            self.seasonality_ = [0]

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
        # Generate forecasts based on the final values of level, trend, and seasonals
        if self.phi == 1:  # No damping case
            phi_h = 1
        else:
            # Geometric series formula for calculating phi + phi^2 + ... + phi^h
            phi_h = self.phi * (1 - self.phi**self.horizon) / (1 - self.phi)
        seasonality = self.seasonality_[
            (self.n_timepoints + self.horizon) % self._seasonal_period
        ]
        fitted_value = self._predict_value(
            self.level_, self.trend_, seasonality, phi_h
        )[0]
        if y is None:
            return np.array([fitted_value])
        else:
            return np.insert(y, 0, fitted_value)[:-1]

    def _predict_value(self, level, trend, seasonality, phi):
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
        if self.trend_type == MULTIPLICATIVE:
            damped_trend = trend**phi
            trend_level_combination = level * damped_trend
        else:  # Additive trend, if no trend, then trend = 0
            damped_trend = trend * phi
            trend_level_combination = level + damped_trend

        # Calculate forecast (fitted value) based on the current components
        if self.seasonality_type == MULTIPLICATIVE:
            fitted_value = trend_level_combination * seasonality
        else:  # Additive seasonality, if no seasonality, then seasonality = 0
            fitted_value = trend_level_combination + seasonality
        return fitted_value, damped_trend, trend_level_combination
