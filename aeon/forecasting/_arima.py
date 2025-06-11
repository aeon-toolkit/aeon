"""ARIMAForecaster.

An implementation of the ARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["ARIMAForecaster"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster
from aeon.utils.optimisation._nelder_mead import nelder_mead


class ARIMAForecaster(BaseForecaster):
    """AutoRegressive Integrated Moving Average (ARIMA) forecaster.

    ARIMA with fixed model structure and fitted parameters found with an
    nelder mead optimizer to minimise the AIC.

    Parameters
    ----------
    p : int, default=1,
        Autoregressive (p) order of the ARIMA model
    d : int, default=0,
        Differencing (d) order of the ARIMA model
    q : int, default=1,
        Moving average (q) order of the ARIMA model
    use_constant: bool = False,
        Presence of a constant/intercept term in the model.

    Attributes
    ----------
    residuals_ : np.ndarray
        Residual errors from the fitted model.
    aic_ : float
        Akaike Information Criterion for the fitted model.
    phi_ : np.ndarray
        Coefficients for the non-seasonal autoregressive terms (length p).
    theta_ : np.ndarray
        Coefficients for the non-seasonal moving average terms (length q).

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
    474.49449...
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(self, p: int = 1, d: int = 0, q: int = 1, use_constant: bool = False):
        self.p = p
        self.d = d
        self.q = q
        self.use_constant = use_constant
        self.phi_ = 0
        self.theta_ = 0
        self._series = []
        self._differenced_series = []
        self.residuals_ = []
        self.fitted_values_ = []
        self.aic_ = 0
        self._model = []
        self._c = 0
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
            Fitted ARIMAForecaster.
        """
        self._series = np.array(y.squeeze(), dtype=np.float64)
        self._model = np.array(
            (1 if self.use_constant else 0, self.p, self.q), dtype=np.int32
        )
        self._differenced_series = np.diff(self._series, n=self.d)
        (self._parameters, self.aic_) = nelder_mead(
            _arima_model_wrapper,
            np.sum(self._model[:3]),
            self._differenced_series,
            self._model,
        )
        (self.c_, self.phi_, self.theta_) = _extract_params(
            self._parameters, self._model
        )
        (self.aic_, self.residuals_, self.fitted_values_) = _arima_model(
            self._parameters,
            _calc_arima,
            self._differenced_series,
            self._model,
            np.empty(0),
        )
        return self

    def _predict(self, y=None, exog=None):
        """
        Predict the next step ahead for training data or y.

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
            Prediction 1 step ahead of the data seen in fit or passed as y.
        """
        if y is not None:
            series = y.squeeze()
        else:
            series = self._series
        n = len(series)
        differenced_series = np.diff(series, n=self.d)
        _, _, predicted_values = _arima_model(
            self._parameters,
            _calc_arima,
            differenced_series,
            self._model,
            self.residuals_,
        )
        # Invert differences
        init = series[n - self.d : n]
        x = np.concatenate((init, predicted_values))
        for _ in range(self.d):
            x = np.cumsum(x)
        return x[self.d :][0]


@njit(cache=True, fastmath=True)
def _aic(residuals, num_params):
    """Calculate the log-likelihood of a model."""
    variance = np.mean(residuals**2)
    liklihood = len(residuals) * (np.log(2 * np.pi) + np.log(variance) + 1)
    return liklihood + 2 * num_params


@njit(fastmath=True)
def _arima_model_wrapper(params, data, model):
    return _arima_model(params, _calc_arima, data, model, np.empty(0))[0]


# Define the ARIMA(p, d, q) likelihood function
@njit(cache=True, fastmath=True)
def _arima_model(params, base_function, data, model, residuals):
    """Calculate the log-likelihood of an ARIMA model given the parameters."""
    formatted_params = _extract_params(params, model)  # Extract parameters

    # Initialize residuals
    n = len(data)
    m = len(residuals)
    num_predictions = n - m + 1
    residuals = np.concatenate((residuals, np.zeros(num_predictions - 1)))
    expect_full_history = m > 0  # I.e. we've been provided with some residuals
    fitted_values = np.zeros(num_predictions)
    for t in range(num_predictions):
        fitted_values[t] = base_function(
            data,
            model,
            m + t,
            formatted_params,
            residuals,
            expect_full_history,
        )
        if t != num_predictions - 1:
            # Only calculate residuals for the predictions we have data for
            residuals[m + t] = data[m + t] - fitted_values[t]
    return _aic(residuals, len(params)), residuals, fitted_values


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
def _calc_arima(data, model, t, formatted_params, residuals, expect_full_history=False):
    """Calculate the ARIMA forecast for time t."""
    if len(model) != 3:
        raise ValueError("Model must be of the form (c, p, q)")
    p = model[1]
    q = model[2]
    if expect_full_history and (t - p < 0 or t - q < 0):
        raise ValueError(
            f"Insufficient data for ARIMA model at time {t}. "
            f"Expected at least {p} past values for AR and {q} for MA."
        )
    # AR part
    phi = formatted_params[1][:p]
    ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])

    # MA part
    theta = formatted_params[2][:q]
    ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])

    c = formatted_params[0][0] if model[0] else 0
    y_hat = c + ar_term + ma_term
    return y_hat
