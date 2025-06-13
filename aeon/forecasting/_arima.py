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

    The model automatically selects the parameters of the model based
    on information criteria, such as AIC.

    Parameters
    ----------
    p : int, default=1,
        Autoregressive (p) order of the ARIMA model
    d : int, default=0,
        Differencing (d) order of the ARIMA model
    q : int, default=1,
        Moving average (q) order of the ARIMA model
    constant_term: bool = False,
        Presence of a constant/intercept term in the model.
    horizon : int, default=1
        The forecasting horizon, i.e., the number of steps ahead to predict.

    Attributes
    ----------
    data_ : np.ndarray
        Original training series values.
    differenced_data_ : np.ndarray
        Differenced version of the training data used for stationarity.
    residuals_ : np.ndarray
        Residual errors from the fitted model.
    aic_ : float
        Akaike Information Criterion for the selected model.
    p, d, q : int
        Parameters passed to the forecaster see p_, d_, q_.
    p_, d_, q_ : int
        Orders of the ARIMA model: autoregressive (p), differencing (d),
        and moving average (q) terms.
    constant_term : bool
        Parameters passed to the forecaster see constant_term_.
    constant_term_ : bool
        Whether to include a constant/intercept term in the model.
    c_ : float
        Estimated constant term (internal use).
    phi_ : np.ndarray
        Coefficients for the non-seasonal autoregressive terms.
    theta_ : np.ndarray
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
    474.49449...
    """

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        constant_term: bool = False,
    ):
        super().__init__(horizon=1, axis=1)
        self.data_ = []
        self.differenced_data_ = []
        self.residuals_ = []
        self.fitted_values_ = []
        self.aic_ = 0
        self.p = p
        self.d = d
        self.q = q
        self.constant_term = constant_term
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.constant_term_ = False
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
        self.model_ = np.array(
            (1 if self.constant_term else 0, self.p, self.q), dtype=np.int32
        )
        self.differenced_data_ = np.diff(self.data_, n=self.d)
        (self.parameters_, self.aic_) = nelder_mead(
            _arima_model_wrapper,
            np.sum(self.model_[:3]),
            self.differenced_data_,
            self.model_,
        )
        (self.c_, self.phi_, self.theta_) = _extract_params(
            self.parameters_, self.model_
        )
        (self.aic_, self.residuals_, self.fitted_values_) = _arima_model(
            self.parameters_,
            _calc_arima,
            self.differenced_data_,
            self.model_,
            np.empty(0),
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
        array[float]
            Predictions len(y) steps ahead of the data seen in fit.
        If y is None, then predict 1 step ahead of the data seen in fit.
        """
        if y is not None:
            combined_data = np.concatenate((self.data_, y.flatten()))
        else:
            combined_data = self.data_
        n = len(self.data_)
        differenced_data = np.diff(combined_data, n=self.d_)
        _aic, _residuals, predicted_values = _arima_model(
            self.parameters_,
            _calc_arima,
            differenced_data,
            self.model_,
            self.residuals_,
        )
        init = combined_data[n - self.d_ : n]
        x = np.concatenate((init, predicted_values))
        for _ in range(self.d_):
            x = np.cumsum(x)
        return x[self.d_ :]


@njit(cache=True, fastmath=True)
def _aic(residuals, num_params):
    """Calculate the log-likelihood of a model."""
    variance = np.mean(residuals**2)
    liklihood = len(residuals) * (np.log(2 * np.pi) + np.log(variance) + 1)
    return liklihood + 2 * num_params


@njit(cache=False, fastmath=True)
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
    phi = formatted_params[1, :p]
    ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])

    # MA part
    theta = formatted_params[2, :q]
    ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])

    c = formatted_params[0, 0] if model[0] else 0
    y_hat = c + ar_term + ma_term
    return y_hat
