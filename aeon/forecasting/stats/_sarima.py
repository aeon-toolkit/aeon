"""Seasonal ARIMA (SARIMA) forecaster."""

__maintainer__ = []
__all__ = ["SARIMA"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._extract_paras import (
    _extract_sarima_params,
    _sarima_coefficients,
    _sarima_lags,
)
from aeon.forecasting.utils._loss_functions import _sarima_model
from aeon.forecasting.utils._nelder_mead import _fit_sarima_parameters


def _get_active_lags(p, q, P, Q, m):
    ar_lags = {i + j * m for i in range(p + 1) for j in range(P + 1)} - {0}
    ma_lags = {i + j * m for i in range(q + 1) for j in range(Q + 1)} - {0}
    return (
        np.array(sorted(ar_lags), dtype=np.int32),
        np.array(sorted(ma_lags), dtype=np.int32),
    )


def _get_diff_poly(d, D, s):
    # Non-seasonal differencing polynomial (1 - B)^d
    poly_d = np.array([1.0], dtype=np.float64)
    for _ in range(d):
        poly_d = np.convolve(poly_d, np.array([1.0, -1.0], dtype=np.float64))

    # Seasonal differencing polynomial (1 - B^s)^D
    step = np.zeros(s + 1, dtype=np.float64)
    step[0] = 1.0
    step[s] = -1.0
    poly_D = np.array([1.0], dtype=np.float64)
    for _ in range(D):
        poly_D = np.convolve(poly_D, step)

    return np.convolve(poly_d, poly_D)


@njit(cache=True, fastmath={"contract"})
def _difference(y, poly_diff):
    n = len(y)
    m = len(poly_diff)
    if n < m:
        return np.empty(0, dtype=np.float64)
    w = np.empty(n - m + 1, dtype=np.float64)
    for i in range(n - m + 1):
        val = 0.0
        for j in range(m):
            val += poly_diff[j] * y[i + m - 1 - j]
        w[i] = val
    return w


@njit(cache=True, fastmath={"contract"})
def _undifference_sarima(w, initial_values, poly_diff):
    m = len(poly_diff)
    n_init = len(initial_values)
    n_out = n_init + len(w)
    out = np.empty(n_out, dtype=np.float64)
    out[:n_init] = initial_values

    for i in range(len(w)):
        val = w[i]
        for j in range(1, m):
            val -= poly_diff[j] * out[n_init + i - j]
        out[n_init + i] = val
    return out


class SARIMA(BaseForecaster, IterativeForecastingMixin):
    """Seasonal AutoRegressive Integrated Moving Average (SARIMA) forecaster.

    The multiplicative model ARIMA(p, d, q)(P, D, Q)m with fixed structure, written
    with sparse AR and MA lag arrays. As in :class:`ARIMA`, the parameters minimise
    the AIC of the conditional sum of squares likelihood, and are restricted to
    the stationary and invertible region. With ``P = D = Q = 0`` the model is
    ``ARIMA(p, d, q)``.

    Parameters
    ----------
    p : int, default=1
        Non-seasonal autoregressive order.
    d : int, default=0
        Non-seasonal differencing order.
    q : int, default=1
        Non-seasonal moving average order.
    P : int, default=0
        Seasonal autoregressive order.
    D : int, default=0
        Seasonal differencing order.
    Q : int, default=0
        Seasonal moving average order.
    seasonal_period : int, default=1
        Number of time points in a season, ``m``.
    use_constant : bool, default=False
        Presence of a constant/intercept term in the model.
    iterations : int, default=200
        Maximum number of Nelder-Mead iterations per optimisation run. The
        parameters are fitted with three runs, from the series mean, from
        Hannan-Rissanen estimates and from the best of those, keeping the best.

    Attributes
    ----------
    residuals_ : np.ndarray
        Residual errors from the fitted model.
    fitted_values_ : np.ndarray
        Fitted values of the differenced series, then its one ahead forecast.
    aic_ : float
        Akaike Information Criterion for the fitted model.
    c_ : float
        Intercept term.
    phi_ : np.ndarray
        Coefficients for non-seasonal autoregressive terms (length p).
    theta_ : np.ndarray
        Coefficients for non-seasonal moving average terms (length q).
    Phi_ : np.ndarray
        Coefficients for seasonal autoregressive terms (length P).
    Theta_ : np.ndarray
        Coefficients for seasonal moving average terms (length Q).

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
        P: int = 0,
        D: int = 0,
        Q: int = 0,
        seasonal_period: int = 1,
        use_constant: bool = False,
        iterations: int = 200,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.seasonal_period = seasonal_period
        self.use_constant = use_constant
        self.iterations = iterations
        self.c_ = 0.0
        self.phi_ = np.empty(0)
        self.theta_ = np.empty(0)
        self.Phi_ = np.empty(0)
        self.Theta_ = np.empty(0)
        self._series = np.empty(0)
        self._differenced_series = np.empty(0)
        self.residuals_ = np.empty(0)
        self.fitted_values_ = np.empty(0)
        self.aic_ = 0.0
        self._model = np.empty(0)
        self._parameters = np.empty(0)
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        if min(self.p, self.d, self.q, self.P, self.D, self.Q) < 0:
            raise ValueError("SARIMA orders must be non-negative.")
        if self.seasonal_period < 1:
            raise ValueError("seasonal_period must be at least 1.")

        self._series = np.array(y.squeeze(), dtype=np.float64)
        ar_lags, ma_lags = _get_active_lags(
            self.p, self.q, self.P, self.Q, self.seasonal_period
        )
        # Model is an array of the (c,p,q,P,Q,m,n_ar_lags,n_ma_lags,lags...)
        self._model = np.array(
            [
                1 if self.use_constant else 0,
                self.p,
                self.q,
                self.P,
                self.Q,
                self.seasonal_period,
                len(ar_lags),
                len(ma_lags),
                *ar_lags,
                *ma_lags,
            ],
            dtype=np.int32,
        )
        self._differenced_series = self._difference_checked(self._series)

        num_params = (1 if self.use_constant else 0) + self.p + self.q + self.P + self.Q
        s = 0.1 / (num_params + 1)
        self._parameters, self.aic_ = _fit_sarima_parameters(
            self._differenced_series,
            self._model,
            max_iter=self.iterations,
            simplex_init=s,
        )
        self.aic_, self.residuals_, self.fitted_values_ = _sarima_model(
            self._parameters,
            self._differenced_series,
            self._model,
        )
        self.c_, self.phi_, self.theta_, self.Phi_, self.Theta_ = (
            _extract_sarima_params(self._parameters, self._model)
        )
        self.forecast_ = self._undifference(
            self.fitted_values_[-1:], self._series
        ).item()
        return self

    def _predict(self, y, exog=None):
        y = np.asarray(y.squeeze(), dtype=np.float64)
        y_diff = self._difference_checked(y)
        fitted_values = _sarima_model(self._parameters, y_diff, self._model)[2]
        return self._undifference(fitted_values[-1:], y).item()

    def _forecast(self, y, exog=None):
        self._fit(y, exog)
        return float(self.forecast_)

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit once on y and recursively forecast multiple steps ahead."""
        y, exog, future_exog = self._check_iterative_forecast_inputs(
            y, prediction_horizon, exog, future_exog
        )
        self.fit(y, exog=exog)
        return self._iterative_forecast_from_fitted(prediction_horizon)

    def _iterative_forecast_from_fitted(self, prediction_horizon):
        """Recursively forecast from the fitted SARIMA state without refitting."""
        h = prediction_horizon
        ar_lags, ma_lags, _ = _sarima_lags(self._model)
        c, ar_values, ma_values = _sarima_coefficients(self._parameters, self._model)
        n = len(self._differenced_series)
        # Future residuals have expectation zero
        residuals = np.zeros(n + h)
        residuals[:n] = self.residuals_
        forecast_series = np.zeros(n + h)
        forecast_series[:n] = self._differenced_series
        for t in range(n, n + h):
            forecast_series[t] = (
                c
                + ar_values @ forecast_series[t - ar_lags]
                + ma_values @ residuals[t - ma_lags]
            )
        return self._undifference(forecast_series[n:], self._series)

    def _difference_checked(self, y):
        """Difference y, checking that a residual can be estimated from it."""
        poly_diff = _get_diff_poly(self.d, self.D, self.seasonal_period)
        if len(y) < len(poly_diff):
            raise ValueError("Series too short for differencing.")
        y_diff = _difference(y, poly_diff)
        if len(y_diff) <= _sarima_lags(self._model)[2]:
            raise ValueError("Series too short for SARIMA with given order.")
        return y_diff

    def _undifference(self, forecasts, y):
        """Undifference forecasts of the differenced series that follows y."""
        poly_diff = _get_diff_poly(self.d, self.D, self.seasonal_period)
        n_init = len(poly_diff) - 1
        if n_init == 0:
            return forecasts
        return _undifference_sarima(forecasts, y[-n_init:], poly_diff)[n_init:]
