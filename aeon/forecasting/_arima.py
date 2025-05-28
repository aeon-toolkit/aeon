"""ARIMAForecaster.

An implementation of the ARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["ARIMAForecaster"]

from math import comb

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster
from aeon.utils.optimisation._nelder_mead import nelder_mead

NOGIL = False
CACHE = True


class ARIMAForecaster(BaseForecaster):
    """AutoRegressive Integrated Moving Average (ARIMA) forecaster.

    The model automatically selects the parameters of the model based
    on information criteria, such as AIC.

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
    constant_term_ : float
        Constant/intercept term in the model.
    c_ : float
        Estimated constant term (internal use).
    phi_ : array-like
        Coefficients for the non-seasonal autoregressive terms.
    theta_ : array-like
        Coefficients for the non-seasonal moving average terms.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice. OTexts, 2014.
       https://otexts.com/fpp3/

    Examples
    --------
    >>> from aeon.forecasting import ARIMAForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ARIMAForecaster(p=2,d=1)
    >>> forecaster.fit(y)
    ARIMAForecaster(d=1, p=2)
    >>> forecaster.predict()
    550.9147246631132
    """

    def __init__(self, p=1, d=0, q=1, constant_term=0, horizon=1):
        super().__init__(horizon=horizon, axis=1)
        self.data_ = []
        self.differenced_data_ = []
        self.residuals_ = []
        self.aic_ = 0
        self.p = p
        self.d = d
        self.q = q
        self.constant_term = constant_term
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.constant_term_ = 0
        self.model_ = []
        self.c_ = 0
        self.phi_ = 0
        self.theta_ = 0
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
        self.p_ = self.p
        self.d_ = self.d
        self.q_ = self.q
        self.constant_term_ = self.constant_term
        self.data_ = np.array(y.squeeze(), dtype=np.float64)
        self.model_ = np.array((self.constant_term, self.p, self.q), dtype=np.int32)
        self.differenced_data_ = np.diff(self.data_, n=self.d)
        (self.parameters_, self.aic_) = nelder_mead(
            _arima_model_wrapper,
            np.sum(self.model_[:3]),
            self.data_,
            self.model_,
        )
        (self.c_, self.phi_, self.theta_) = _extract_params(
            self.parameters_, self.model_
        )
        (self.aic_, self.residuals_) = _arima_model(
            self.parameters_, _calc_arima, self.differenced_data_, self.model_
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
        value = _calc_arima(
            self.differenced_data_,
            self.model_,
            len(self.differenced_data_),
            _extract_params(self.parameters_, self.model_),
            self.residuals_,
        )
        history = self.data_[::-1]
        # Step 2: undo ordinary differencing
        for k in range(1, self.d_ + 1):
            value += (-1) ** (k + 1) * comb(self.d_, k) * history[k - 1]
        return float(value)


@njit(cache=True, fastmath=True)
def _aic(residuals, num_params):
    """Calculate the log-likelihood of a model."""
    variance = np.mean(residuals**2)
    liklihood = len(residuals) * (np.log(2 * np.pi) + np.log(variance) + 1)
    return liklihood + 2 * num_params


@njit(fastmath=True)
def _arima_model_wrapper(params, data, model):
    return _arima_model(params, _calc_arima, data, model)[0]


# Define the ARIMA(p, d, q) likelihood function
@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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
    starts = np.cumsum(np.concatenate((np.zeros(1, dtype=np.int32), model[:-1])))
    n = len(starts)
    max_len = np.max(model)
    result = np.full((n, max_len), np.nan, dtype=params.dtype)
    for i in range(n):
        length = model[i]
        start = starts[i]
        result[i, :length] = params[start : start + length]
    return result


@njit(cache=True, fastmath=True)
def _calc_arima(data, model, t, formatted_params, residuals):
    """Calculate the ARIMA forecast for time t."""
    if len(model) != 3:
        raise ValueError("Model must be of the form (c, p, q)")
    # AR part
    p = model[1]
    phi = formatted_params[1][:p]
    ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])

    # MA part
    q = model[2]
    theta = formatted_params[2][:q]
    ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])

    c = formatted_params[0][0] if model[0] else 0
    y_hat = c + ar_term + ma_term
    return y_hat
