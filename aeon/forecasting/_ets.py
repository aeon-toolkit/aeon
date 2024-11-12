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

from aeon.forecasting.base import BaseForecaster

NONE = 0
ADDITIVE = 1
MULTIPLICATIVE = 2


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
        assert error_type != NONE, "Error must be either additive or multiplicative"
        if seasonal_period < 1 or seasonality_type == NONE:
            seasonal_period = 1
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period


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

    default_model_type = ModelType()

    def __init__(
        self,
        model_type=default_model_type,
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
        self.season_ = None
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
        data = y.squeeze()
        self.n_timepoints = len(data)
        self._initialise(data)
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        mul_liklihood_pt2 = 0
        self.residuals_ = np.zeros(
            self.n_timepoints
        )  # 1 Less residual than data points
        for t, data_item in enumerate(data[self.model_type.seasonal_period :]):
            # Calculate level, trend, and seasonal components
            fitted_value, error = self._update_states(
                data_item, t % self.model_type.seasonal_period
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
        model = self.model_type
        # Retrieve the current state values
        level = self.level_
        trend = self.trend_
        seasonality = self.season_[seasonal_index]
        fitted_value, damped_trend, trend_level_combination = self._predict_value(
            trend, level, seasonality, self.phi
        )
        # Calculate the error term (observed value - fitted value)
        if model.error_type == MULTIPLICATIVE:
            error = data_item / fitted_value - 1  # Multiplicative error
        else:
            error = data_item - fitted_value  # Additive error
        # Update level
        if model.error_type == MULTIPLICATIVE:
            self.level_ = trend_level_combination * (1 + self.alpha * error)
            self.trend_ = damped_trend * (1 + self.beta * error)
            self.season_[seasonal_index] = seasonality * (1 + self.gamma * error)
            if model.seasonality_type == ADDITIVE:
                self.level_ += (
                    self.alpha * error * seasonality
                )  # Add seasonality correction
                self.season_[seasonal_index] += (
                    self.gamma * error * trend_level_combination
                )
                if model.trend_type == ADDITIVE:
                    self.trend_ += (level + seasonality) * self.beta * error
                else:
                    self.trend_ += seasonality / level * self.beta * error
            elif model.trend_type == ADDITIVE:
                self.trend_ += level * self.beta * error
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
            self.level_ = (
                trend_level_combination + self.alpha * error / level_correction
            )
            self.trend_ = damped_trend + self.beta * error / trend_correction
            self.season_[seasonal_index] = (
                seasonality + self.gamma * error / seasonality_correction
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
        model = self.model_type
        # Initial Level: Mean of the first season
        self.level_ = np.mean(data[: model.seasonal_period])
        # Initial Trend
        if model.trend_type == ADDITIVE:
            # Average difference between corresponding points in the first two seasons
            self.trend_ = np.mean(
                data[model.seasonal_period : 2 * model.seasonal_period]
                - data[: model.seasonal_period]
            )
        elif model.trend_type == MULTIPLICATIVE:
            # Average ratio between corresponding points in the first two seasons
            self.trend_ = np.mean(
                data[model.seasonal_period : 2 * model.seasonal_period]
                / data[: model.seasonal_period]
            )
        else:
            # No trend
            self.trend_ = 0
            self.beta = (
                0  # Required for the equations in _update_states to work correctly
            )
        # Initial Seasonality
        if model.seasonality_type == ADDITIVE:
            # Seasonal component is the difference
            # from the initial level for each point in the first season
            self.season_ = data[: model.seasonal_period] - self.level_
        elif model.seasonality_type == MULTIPLICATIVE:
            # Seasonal component is the ratio of each point in the first season
            # to the initial level
            self.season_ = data[: model.seasonal_period] / self.level_
        else:
            # No seasonality
            self.season_ = [0]
            self.gamma = (
                0  # Required for the equations in _update_states to work correctly
            )

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
            phi_h = float(self.horizon)
        else:
            # Geometric series formula for calculating phi + phi^2 + ... + phi^h
            phi_h = self.phi * (1 - self.phi**self.horizon) / (1 - self.phi)
        seasonality = self.season_[
            (self.n_timepoints + self.horizon) % self.model_type.seasonal_period
        ]
        fitted_value = self._predict_value(
            self.trend_, self.level_, seasonality, phi_h
        )[0]
        return fitted_value

    def _predict_value(self, trend, level, seasonality, phi):
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
        model = self.model_type
        # Apply damping parameter and
        # calculate commonly used combination of trend and level components
        if model.trend_type == MULTIPLICATIVE:
            damped_trend = trend**phi
            trend_level_combination = level * damped_trend
        else:  # Additive trend, if no trend, then trend = 0
            damped_trend = trend * phi
            trend_level_combination = level + damped_trend

        # Calculate forecast (fitted value) based on the current components
        if model.seasonality_type == MULTIPLICATIVE:
            fitted_value = trend_level_combination * seasonality
        else:  # Additive seasonality, if no seasonality, then seasonality = 0
            fitted_value = trend_level_combination + seasonality
        return fitted_value, damped_trend, trend_level_combination
