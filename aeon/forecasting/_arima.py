"""ARIMAForecaster.

An implementation of the ARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["ARIMAForecaster"]

from math import comb

import numpy as np

from aeon.forecasting.base import BaseForecaster
from aeon.utils.forecasting._hypo_tests import kpss_test
from aeon.utils.forecasting._seasonality import calc_seasonal_period
from aeon.utils.optimisation._nelder_mead import nelder_mead

NOGIL = False
CACHE = True


class ARIMAForecaster(BaseForecaster):
    """AutoRegressive Integrated Moving Average (ARIMA) forecaster.

    Implements the Hyndman-Khandakar automatic ARIMA algorithm for time series
    forecasting with optional seasonal components. The model automatically selects
    the orders of the non-seasonal (p, d, q) and seasonal (P, D, Q, m) components
    based on information criteria, such as AIC.

    Parameters
    ----------
    horizon : int, default=1
        The forecasting horizon, i.e., the number of steps ahead to predict.

    Attributes
    ----------
    data_ : list of float
        Original training series values.
    differenced_data_ : list of float
        Differenced version of the training data used for stationarity.
    residuals_ : list of float
        Residual errors from the fitted model.
    aic_ : float
        Akaike Information Criterion for the selected model.
    p_, d_, q_ : int
        Orders of the ARIMA model: autoregressive (p), differencing (d),
        and moving average (q) terms.
    ps_, ds_, qs_ : int
        Orders of the seasonal ARIMA model: seasonal AR (P), seasonal differencing (D),
        and seasonal MA (Q) terms.
    seasonal_period_ : int
        Length of the seasonal cycle.
    constant_term_ : float
        Constant/intercept term in the model.
    c_ : float
        Estimated constant term (internal use).
    phi_ : array-like
        Coefficients for the non-seasonal autoregressive terms.
    phi_s_ : array-like
        Coefficients for the seasonal autoregressive terms.
    theta_ : array-like
        Coefficients for the non-seasonal moving average terms.
    theta_s_ : array-like
        Coefficients for the seasonal moving average terms.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice. OTexts, 2014.
       https://otexts.com/fpp3/
    """

    def __init__(self, horizon=1):
        super().__init__(horizon=horizon, axis=1)
        self.data_ = []
        self.differenced_data_ = []
        self.residuals_ = []
        self.aic_ = 0
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.ps_ = 0
        self.ds_ = 0
        self.qs_ = 0
        self.seasonal_period_ = 0
        self.constant_term_ = 0
        self.model_ = []
        self.c_ = 0
        self.phi_ = 0
        self.phi_s_ = 0
        self.theta_ = 0
        self.theta_s_ = 0
        self.parameters_ = []

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
        self.data_ = np.array(y.squeeze(), dtype=np.float64)
        (
            self.differenced_data_,
            self.d_,
            self.ds_,
            self.model_,
            self.parameters_,
            self.aic_,
        ) = _auto_arima(self.data_)
        (
            self.constant_term_,
            self.p_,
            self.q_,
            self.ps_,
            self.qs_,
            self.seasonal_period_,
        ) = self.model_
        (self.c_, self.phi_, self.phi_s_, self.theta_, self.theta_s_) = _extract_params(
            self.parameters_, self.model_
        )
        (
            self.aic_,
            self.residuals_,
        ) = _arima_model(
            self.parameters_, _calc_sarima, self.differenced_data_, self.model_
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
        value = _calc_sarima(
            self.differenced_data_,
            self.model_,
            len(self.differenced_data_),
            _extract_params(self.parameters_, self.model_),
            self.residuals_,
        )
        history = self.data_[::-1]
        differenced_history = np.diff(self.data_, n=self.d_)[::-1]
        # Step 1: undo seasonal differencing on y^(d)
        for k in range(1, self.ds_ + 1):
            lag = k * self.seasonal_period_
            value += (-1) ** (k + 1) * comb(self.ds_, k) * differenced_history[lag - 1]

        # Step 2: undo ordinary differencing
        for k in range(1, self.d_ + 1):
            value += (-1) ** (k + 1) * comb(self.d_, k) * history[k - 1]
        return value


def _aic(residuals, num_params):
    """Calculate the log-likelihood of a model."""
    variance = np.mean(residuals**2)
    liklihood = len(residuals) * (np.log(2 * np.pi) + np.log(variance) + 1)
    return liklihood + 2 * num_params


# Define the ARIMA(p, d, q) likelihood function
def _arima_model(params, base_function, data, model):
    """Calculate the log-likelihood of an ARIMA model given the parameters."""
    formatted_params = _extract_params(params, model)  # Extract parameters

    # Initialize residuals
    n = len(data)
    residuals = np.zeros(n)
    for t in range(n):
        y_hat = base_function(
            data,
            model,
            t,
            formatted_params,
            residuals,
        )
        residuals[t] = data[t] - y_hat
    return _aic(residuals, len(params)), residuals


# Define the SARIMA(p, d, q)(P, D, Q) likelihood function


def _extract_params(params, model):
    """Extract ARIMA parameters from the parameter vector."""
    if len(params) != np.sum(model):
        previous_length = np.sum(model)
        model = model[:-1]  # Remove the seasonal period
        if len(params) != np.sum(model):
            raise ValueError(
                f"Expected {previous_length} parameters for a non-seasonal model or \
                    {np.sum(model)} parameters for a seasonal model, got {len(params)}"
            )
    starts = np.cumsum([0] + model[:-1])
    return [params[s : s + l].tolist() for s, l in zip(starts, model)]


def _calc_arima(data, model, t, formatted_params, residuals):
    """Calculate the ARIMA forecast for time t."""
    if len(model) != 3:
        raise ValueError("Model must be of the form (c, p, q)")
    # AR part
    p = model[1]
    phi = formatted_params[1]
    ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])

    # MA part
    q = model[2]
    theta = formatted_params[2]
    ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])

    c = formatted_params[0][0] if model[0] else 0
    y_hat = c + ar_term + ma_term
    return y_hat


def _calc_sarima(data, model, t, formatted_params, residuals):
    """Calculate the SARIMA forecast for time t."""
    if len(model) != 6:
        raise ValueError("Model must be of the form (c, p, q, ps, qs, seasonal_period)")
    arima_forecast = _calc_arima(data, model[:3], t, formatted_params, residuals)
    seasonal_period = model[5]
    # Seasonal AR part
    ps = model[3]
    phi_s = formatted_params[3]
    ars_term = (
        0
        if (t - seasonal_period * ps) < 0
        else np.dot(phi_s, data[t - seasonal_period * ps : t : seasonal_period][::-1])
    )
    # Seasonal MA part
    qs = model[4]
    theta_s = formatted_params[4]
    mas_term = (
        0
        if (t - seasonal_period * qs) < 0
        else np.dot(
            theta_s, residuals[t - seasonal_period * qs : t : seasonal_period][::-1]
        )
    )
    return arima_forecast + ars_term + mas_term


def make_arima_llf(base_function, data, model):
    """
    Return a parameterized log-likelihood function for ARIMA.

    This can then be used with an optimization algorithm.
    """

    def loss_fn(v):
        return _arima_model(v, base_function, data, model)[0]

    return loss_fn


def _auto_arima(data):
    """
    Implement the Hyndman-Khandakar algorithm.

    For automatic ARIMA model selection.
    """
    seasonal_period = calc_seasonal_period(data)
    difference = 0
    while not kpss_test(data)[1]:
        data = np.diff(data, n=1)
        difference += 1
    seasonal_difference = 1 if seasonal_period > 1 else 0
    if seasonal_difference:
        data = data[seasonal_period:] - data[:-seasonal_period]
    include_c = 1 if difference == 0 else 0
    model_parameters = [
        [include_c, 2, 2, 0, 0, seasonal_period],
        [include_c, 0, 0, 0, 0, seasonal_period],
        [include_c, 1, 0, 0, 0, seasonal_period],
        [include_c, 0, 1, 0, 0, seasonal_period],
    ]
    model_points = []
    model_scores = []
    for p in model_parameters:
        points, aic = nelder_mead(make_arima_llf(_calc_sarima, data, p), np.sum(p[:5]))
        model_points.append(points)
        model_scores.append(aic)
    best_score = min(model_scores)
    best_index = model_scores.index(best_score)
    current_model = model_parameters[best_index]
    current_points = model_points[best_index]
    while True:
        better_model = False
        for param_no in range(1, 5):
            for adjustment in [-1, 1]:
                if (current_model[param_no] + adjustment) < 0:
                    continue
                model = current_model.copy()
                model[param_no] += adjustment
                for constant_term in [0, 1]:
                    model[0] = constant_term
                    points, aic = nelder_mead(
                        make_arima_llf(_calc_sarima, data, model), np.sum(model[:5])
                    )
                    if aic < best_score:
                        current_model = model.copy()
                        current_points = points
                        best_score = aic
                        better_model = True
        if not better_model:
            break
    return (
        data,
        difference,
        seasonal_difference,
        current_model,
        current_points,
        best_score,
    )
