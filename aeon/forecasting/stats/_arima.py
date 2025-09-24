"""ARIMA.

An implementation of the ARIMA forecasting algorithm,
including automatic hyperparameter tuning version
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["ARIMA", "AutoARIMA"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._extract_paras import _extract_arma_params
from aeon.forecasting.utils._hypo_tests import kpss_test
from aeon.forecasting.utils._nelder_mead import nelder_mead
from aeon.forecasting.utils._undifference import _undifference


class ARIMA(BaseForecaster, IterativeForecastingMixin):
    """AutoRegressive Integrated Moving Average (ARIMA) forecaster.

    ARIMA with fixed model structure and fitted parameters found with an
    Nelder Mead optimizer to minimise the AIC.

    Parameters
    ----------
    p : int, default=1,
        Autoregressive (p) order of the ARIMA model
    d : int, default=0,
        Differencing (d) order of the ARIMA model
    q : int, default=1,
        Moving average (q) order of the ARIMA model
    use_constant : bool = False,
        Presence of a constant/intercept term in the model.
    grid : int: 0
    iterations : int, default = 200
        Maximum number of iterations to use in the Nelder-Mead parameter search.

    Attributes
    ----------
    residuals_ : np.ndarray
        Residual errors from the fitted model.
    aic_ : float
        Akaike Information Criterion for the fitted model.
    c_ : float, default = 0
        Intercept term.
    phi_ : np.ndarray
        Coefficients for autoregressive terms (length p).
    theta_ : np.ndarray
        Coefficients for moving average terms (length q).

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice. OTexts, 2014.
       https://otexts.com/fpp3/
    """

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        use_constant: bool = False,
        iterations: int = 200,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.use_constant = use_constant
        self.iterations = iterations
        self.phi_ = 0
        self.theta_ = 0
        self.c_ = 0
        self._series = []
        self._differenced_series = []
        self.residuals_ = []
        self.fitted_values_ = []
        self.aic_ = 0
        self._model = []
        self._parameters = []
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit ARIMA forecaster to series y to predict one ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        exog : np.ndarray, default =None
            Not allowed for this forecaster

        Returns
        -------
        self
            Fitted ARIMA.
        """
        self._series = np.array(y.squeeze(), dtype=np.float64)
        # Model is an array of the (c,p,q)
        self._model = np.array(
            (1 if self.use_constant else 0, self.p, self.q), dtype=np.int32
        )
        self._differenced_series = np.diff(self._series, n=self.d)
        s = 0.1 / (np.sum(self._model) + 1)  # Randomise
        # Nelder Mead returns the parameters in a single array
        (self._parameters, self.aic_) = nelder_mead(
            0,
            np.sum(self._model[:3]),
            self._differenced_series,
            self._model,
            max_iter=self.iterations,
            simplex_init=s,
        )
        #
        (self.aic_, self.residuals_, self.fitted_values_) = _arima_model(
            self._parameters,
            self._differenced_series,
            self._model,
        )
        formatted_params = _extract_arma_params(
            self._parameters, self._model
        )  # Extract
        # parameters
        differenced_forecast = self.fitted_values_[-1]

        if self.d == 0:
            self.forecast_ = differenced_forecast
        else:
            self.forecast_ = _undifference(
                np.array([differenced_forecast]), self._series[-self.d :]
            )[self.d]
        if self.use_constant:
            self.c_ = formatted_params[0][0]
        self.phi_ = formatted_params[1][: self.p]
        self.theta_ = formatted_params[2][: self.q]

        return self

    def _predict(self, y, exog=None):
        """
        Predict the next step ahead for y.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the value of. y can be independent of the series
            seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            Prediction 1 step ahead of the last value in y.
        """
        y = y.squeeze()
        p, q, d = self.p, self.q, self.d
        phi, theta = self.phi_, self.theta_
        c = 0.0
        if self.use_constant:
            c = self.c_

        # Apply differencing
        if d > 0:
            if len(y) <= d:
                raise ValueError("Series too short for differencing.")
            y_diff = np.diff(y, n=d)
        else:
            y_diff = y

        n = len(y_diff)
        if n < max(p, q):
            raise ValueError("Series too short for ARMA(p,q) with given order.")

        # Estimate in-sample residuals using model (fixed parameters)
        residuals = np.zeros(n)
        for t in range(max(p, q), n):
            ar_part = np.dot(phi, y_diff[t - np.arange(1, p + 1)]) if p > 0 else 0.0
            ma_part = (
                np.dot(theta, residuals[t - np.arange(1, q + 1)]) if q > 0 else 0.0
            )
            pred = c + ar_part + ma_part
            residuals[t] = y_diff[t] - pred

        # Use most recent p values of y_diff and q values of residuals to forecast t+1
        ar_forecast = np.dot(phi, y_diff[-np.arange(1, p + 1)]) if p > 0 else 0.0
        ma_forecast = np.dot(theta, residuals[-np.arange(1, q + 1)]) if q > 0 else 0.0

        forecast_diff = c + ar_forecast + ma_forecast

        # Undifference the forecast
        if self.d == 0:
            return forecast_diff
        else:
            return _undifference(np.array([forecast_diff]), self._series[-self.d :])[
                self.d
            ]

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return float(self.forecast_)

    def iterative_forecast(self, y, prediction_horizon):
        self.fit(y)
        n = len(self._differenced_series)
        p, q = self.p, self.q
        phi, theta = self.phi_, self.theta_
        h = prediction_horizon
        c = 0.0
        if self.use_constant:
            c = self.c_

        # Start with a copy of the original series and residuals
        residuals = np.zeros(len(self.residuals_) + h)
        residuals[: len(self.residuals_)] = self.residuals_
        forecast_series = np.zeros(n + h)
        forecast_series[:n] = self._differenced_series
        for i in range(h):
            # Get most recent p values (lags)
            t = n + i
            ar_term = 0.0
            if p > 0:
                ar_term = np.dot(phi, forecast_series[t - np.arange(1, p + 1)])
            # Get most recent q residuals (lags)
            ma_term = 0.0
            if q > 0:
                ma_term = np.dot(theta, residuals[t - np.arange(1, q + 1)])
            next_value = c + ar_term + ma_term
            # Append prediction and a zero residual (placeholder)
            forecast_series[n + i] = next_value
            # Can't compute real residual during prediction, leave as zero

        # Correct differencing using forecast values
        y_forecast_diff = forecast_series[n : n + h]
        if self.d == 0:
            return y_forecast_diff
        else:
            return _undifference(y_forecast_diff, self._series[-self.d :])[self.d :]


class AutoARIMA(BaseForecaster, IterativeForecastingMixin):
    """AutoRegressive Integrated Moving Average (ARIMA) forecaster.

    Implements the Hyndman-Khandakar automatic ARIMA algorithm for time series
    forecasting with optional seasonal components. The model automatically selects
    the orders of the (p, d, q) components based on information criteria, such as AIC.

    Parameters
    ----------
    max_p : int, default=3
        Maximum order of the autoregressive (AR) component to consider when
        searching for the best ARIMA model.
    max_d : int, default=3
        Maximum order of differencing (number of times the series may be
        differenced) considered to achieve stationarity.
    max_q : int, default=2
        Maximum order of the moving-average (MA) component to consider when
        searching for the best ARIMA model.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice. OTexts, 2014.
       https://otexts.com/fpp3/

    Examples
    --------
    >>> from aeon.forecasting.stats import AutoARIMA
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoARIMA()
    >>> forecaster.forecast(y)
    481.87157356139943
    """

    def __init__(self, max_p=3, max_d=3, max_q=2):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.constant_term_ = False
        self.final_model_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit AutoARIMA forecaster to series y.

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
            Fitted ARIMAForecaster.
        """
        series = np.array(y.squeeze(), dtype=np.float64)
        differenced_series = series.copy()
        self.d_ = 0
        while not kpss_test(differenced_series)[1] and self.d_ <= self.max_d:
            differenced_series = np.diff(differenced_series, n=1)
            self.d_ += 1
        include_constant = 1 if self.d_ == 0 else 0
        initial_parameters = [[include_constant, 0, 0]]
        if self.max_p >= 1:
            initial_parameters.append([include_constant, 1, 0])
        if self.max_q >= 1:
            initial_parameters.append([include_constant, 0, 1])
        if (self.max_p >= 2) and (self.max_q >= 2):
            initial_parameters.append([include_constant, 2, 2])
        model_parameters = np.array(initial_parameters)
        parameter_limits = np.array([self.max_p, self.max_q])
        (
            model,
            _,
            self.fit_aic_,
        ) = _auto_arima(differenced_series, 0, model_parameters, 3, parameter_limits)
        (
            constant_term_int,
            self.p_,
            self.q_,
        ) = model
        self.constant_term_ = constant_term_int == 1
        self.final_model_ = ARIMA(self.p_, self.d_, self.q_, self.constant_term_)
        self.final_model_.fit(y)
        self.forecast_ = self.final_model_.forecast_
        return self

    def _predict(self, y, exog=None):
        """
        Predict the next step ahead for y.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the value of. y can be independent of the series
            seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            Prediction 1 step ahead of the last value in y.
        """
        return self.final_model_.predict(y, exog)

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self._fit(y, exog)
        return float(self.final_model_.forecast_)

    def iterative_forecast(self, y, prediction_horizon):
        """Forecast ``prediction_horizon`` prediction using a single model fit on `y`.

        This function implements the iterative forecasting strategy (also called
        recursive or iterated). This involves a single model fit on ``y`` which is then
        used to make ``prediction_horizon`` ahead forecasts using its own predictions as
        inputs for future forecasts. This is done by taking the prediction at step
        ``i`` and feeding it back into the model to help predict for step ``i+1``.
        The basic contract of `iterative_forecast` is that `fit` is only ever called
        once.

        y : np.ndarray
            The time series to make forecasts about.  Must be of shape
            ``(n_channels, n_timepoints)`` if a multivariate time series.
        prediction_horizon : int
            The number of future time steps to forecast.

        Returns
        -------
        np.ndarray
            An array of shape `(prediction_horizon,)` containing the forecasts for
            each horizon.

        Raises
        ------
        ValueError
            if prediction_horizon` less than 1.
        """
        return self.final_model_.iterative_forecast(y, prediction_horizon)


@njit(cache=True, fastmath=True)
def _aic(residuals, num_params):
    """Calculate the log-likelihood of a model."""
    variance = np.mean(residuals**2)
    likelihood = len(residuals) * (np.log(2 * np.pi) + np.log(variance) + 1)
    return likelihood + 2 * num_params


# Define the ARIMA(p, d, q) likelihood function
@njit(cache=True, fastmath=True)
def _arima_model(params, data, model):
    """Calculate the log-likelihood of an ARIMA model given the parameters."""
    formatted_params = _extract_arma_params(params, model)  # Extract parameters

    # Initialize residuals
    n = len(data)
    num_predictions = n + 1
    residuals = np.zeros(num_predictions - 1)
    fitted_values = np.zeros(num_predictions)
    # Leave first max(p,q) residuals and fitted as zero.
    for t in range(max(model[1], model[2]), num_predictions):
        fitted_values[t] = _in_sample_forecast(
            data, model, t, formatted_params, residuals
        )
        if t != num_predictions - 1:
            # Only calculate residuals for the predictions we have data for
            residuals[t] = data[t] - fitted_values[t]
    return _aic(residuals, len(params)), residuals, fitted_values


@njit(cache=True, fastmath=True)
def _in_sample_forecast(data, model, t, formatted_params, residuals):
    """Efficient ARMA one-step forecast at time t for fitted model."""
    p = model[1]
    q = model[2]
    c = formatted_params[0][0] if model[0] else 0.0

    ar_term = 0.0
    for j in range(min(p, t)):
        ar_term += formatted_params[1, j] * data[t - j - 1]

    ma_term = 0.0
    for j in range(min(q, t)):
        ma_term += formatted_params[2, j] * residuals[t - j - 1]

    return c + ar_term + ma_term


@njit(cache=True, fastmath=True)
def _auto_arima(
    differenced_data,
    loss_id,
    initial_model_parameters,
    num_model_params=3,
    parameter_limits=None,
):
    """
    Implement the Hyndman-Khandakar algorithm.

    For automatic ARIMA model selection.
    """
    best_score = -1
    best_model = initial_model_parameters[0]
    best_points = None
    for model in initial_model_parameters:
        s = 0.1 / (np.sum(model) + 1)
        points, aic = nelder_mead(
            loss_id,
            np.sum(model[:num_model_params]),
            differenced_data,
            model,
            max_iter=200,
            simplex_init=s,
        )
        if (aic < best_score) or (best_score == -1):
            best_score = aic
            best_model = model
            best_points = points

    while True:
        better_model = False
        for param_no in range(1, num_model_params):
            for adjustment in [-1, 1]:
                if (best_model[param_no] + adjustment) < 0 or (
                    best_model[param_no] + adjustment
                ) > parameter_limits[param_no - 1]:
                    continue
                model = best_model.copy()
                model[param_no] += adjustment
                for constant_term in [0, 1]:
                    model[0] = constant_term
                    points, aic = nelder_mead(
                        loss_id,
                        np.sum(model[:num_model_params]),
                        differenced_data,
                        model,
                    )
                    if aic < best_score:
                        best_model = model.copy()
                        best_points = points
                        best_score = aic
                        better_model = True
        if not better_model:
            break
    return (
        best_model,
        best_points,
        best_score,
    )
