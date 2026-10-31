"""Seasonal ARIMA (SARIMA) forecaster."""

__maintainer__ = []
__all__ = ["SARIMA", "AutoSARIMA"]

import time

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._extract_paras import (
    _extract_sarima_params,
    _sarima_coefficients,
    _sarima_lags,
)
from aeon.forecasting.utils._hypo_tests import kpss_test
from aeon.forecasting.utils._loss_functions import _is_stationary_ar, _sarima_model
from aeon.forecasting.utils._nelder_mead import _fit_sarima_parameters
from aeon.transformations.series._stl import STLSeriesTransformer

SEASONAL_STRENGTH_THRESHOLD = 0.64


@njit(cache=True, fastmath={"contract"})
def _clear_of_unit_circle(coefs, margin):
    """Check that all roots of ``1 - sum(coefs[i] z^(i + 1))`` exceed ``margin``."""
    scaled = np.empty(len(coefs))
    for i in range(len(coefs)):
        scaled[i] = coefs[i] * margin ** (i + 1)
    return _is_stationary_ar(scaled)


@njit(cache=True, fastmath={"contract"})
def _roots_clear_of_unit_circle(params, model, margin=1.01):
    """Check that all AR and MA roots have modulus greater than ``margin``.

    A root ``w`` of a seasonal polynomial in ``B^m`` gives roots of modulus
    ``|w|^(1 / m)`` in ``B``, so it has to clear ``margin^m``.
    """
    _, phi, theta, Phi, Theta = _extract_sarima_params(params, model)
    seasonal_margin = margin ** model[5]
    return (
        _clear_of_unit_circle(phi, margin)
        and _clear_of_unit_circle(-theta, margin)
        and _clear_of_unit_circle(Phi, seasonal_margin)
        and _clear_of_unit_circle(-Theta, seasonal_margin)
    )


def _get_active_lags(p, q, P, Q, m):
    ar_lags = {i + j * m for i in range(p + 1) for j in range(P + 1)} - {0}
    ma_lags = {i + j * m for i in range(q + 1) for j in range(Q + 1)} - {0}
    return (
        np.array(sorted(ar_lags), dtype=np.int32),
        np.array(sorted(ma_lags), dtype=np.int32),
    )


def _build_model(use_constant, p, q, P, Q, m):
    """Return the array ``[c, p, q, P, Q, m, n_ar_lags, n_ma_lags, lags...]``."""
    ar_lags, ma_lags = _get_active_lags(p, q, P, Q, m)
    return np.array(
        [
            1 if use_constant else 0,
            p,
            q,
            P,
            Q,
            m,
            len(ar_lags),
            len(ma_lags),
            *ar_lags,
            *ma_lags,
        ],
        dtype=np.int32,
    )


def _fit_candidate(data, model, max_iter):
    """Fit one SARIMA model to the differenced ``data``."""
    num_params = model[0] + model[1] + model[2] + model[3] + model[4]
    s = 0.1 / (num_params + 1)
    return _fit_sarima_parameters(data, model, max_iter=max_iter, simplex_init=s)


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
        return self._fit_with_parameters(y)

    def _fit_with_parameters(self, y, parameters=None):
        """Fit as ``_fit``, optionally with parameters already estimated.

        ``parameters``, laid out as in ``_extract_sarima_params``, must have been
        estimated on this ``y`` differenced ``d`` and ``D`` times. If None they
        are estimated here.
        """
        if min(self.p, self.d, self.q, self.P, self.D, self.Q) < 0:
            raise ValueError("SARIMA orders must be non-negative.")
        if self.seasonal_period < 1:
            raise ValueError("seasonal_period must be at least 1.")

        self._series = np.array(y.squeeze(), dtype=np.float64)
        self._model = _build_model(
            self.use_constant, self.p, self.q, self.P, self.Q, self.seasonal_period
        )
        self._differenced_series = self._difference_checked(self._series)

        if parameters is None:
            self._parameters, self.aic_ = _fit_candidate(
                self._differenced_series, self._model, self.iterations
            )
        else:
            self._parameters = np.asarray(parameters, dtype=np.float64)
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


def _seasonal_strength(y, m):
    """Strength of seasonality of ``y`` from an STL decomposition, in [0, 1].

    ``1 - var(remainder) / var(remainder + seasonal)``, as ``seas.heuristic`` in
    R's forecast package, with its seasonal window of 11. Series of at most two
    seasons are too short to decompose and have strength 0.
    """
    if len(y) <= 2 * m:
        return 0.0
    components = STLSeriesTransformer(
        period=m, seasonal=11, output="all"
    ).fit_transform(y)
    seasonal, remainder = components[:, 0], components[:, 2]
    detrended_var = np.var(remainder + seasonal, ddof=1)
    if not detrended_var > 0:
        return 0.0
    return float(np.clip(1 - np.var(remainder, ddof=1) / detrended_var, 0.0, 1.0))


def _select_differencing(y, m, max_d, max_D):
    """Choose ``d`` and ``D``.

    ``D`` from the strength of seasonality, with threshold 0.64 as ``nsdiffs`` in
    R's forecast package, then ``d`` as :class:`AutoARIMA` does, from KPSS tests
    with ``trunc(3 sqrt(n) / 13)`` lags, here on the seasonally differenced series.
    """
    D = 0
    if m > 1:
        while D < max_D and _seasonal_strength(y, m) >= SEASONAL_STRENGTH_THRESHOLD:
            y = y[m:] - y[:-m]
            D += 1
    d = 0
    while (
        d < max_d
        and len(y) > 1
        and not kpss_test(y, lags=int(3 * np.sqrt(len(y)) / 13))[1]
    ):
        y = np.diff(y)
        d += 1
    return d, D


class AutoSARIMA(BaseForecaster, IterativeForecastingMixin):
    """Seasonal ARIMA forecaster with automatic order selection.

    The stepwise Hyndman-Khandakar algorithm [1]_ of :class:`AutoARIMA`, searching
    over the seasonal orders as well. ``D`` is chosen from the strength of
    seasonality and ``d`` from KPSS tests. The search starts from the seasonal
    forms of the :class:`AutoARIMA` starting models, ``(p, q)(P, Q)`` of
    ``(0, 0)(0, 0)``, ``(1, 0)(1, 0)``, ``(0, 1)(0, 1)`` and ``(2, 2)(1, 1)``, with
    a constant if ``d + D == 0``. It then moves to the first better neighbour
    until there is none. Neighbours change ``P`` and ``Q``, or ``p`` and ``q``, by
    at most one each, with and without a constant.

    Models are compared by the AIC of the conditional sum of squares likelihood
    and a fit with an AR or MA root within 1.01 of the unit circle is rejected.
    With ``seasonal_period=1`` this is the search of :class:`AutoARIMA`.

    Parameters
    ----------
    seasonal_period : int, default=1
        Number of time points in a season. With 1 the model is non-seasonal.
    max_p : int, default=3
        Maximum non-seasonal autoregressive order.
    max_d : int, default=3
        Maximum non-seasonal differencing order.
    max_q : int, default=2
        Maximum non-seasonal moving average order.
    max_P : int, default=2
        Maximum seasonal autoregressive order.
    max_D : int, default=1
        Maximum seasonal differencing order.
    max_Q : int, default=2
        Maximum seasonal moving average order.

    Attributes
    ----------
    p_, d_, q_, P_, D_, Q_ : int
        The selected orders.
    constant_term_ : bool
        Whether the selected model has a constant.
    final_model_ : SARIMA
        The selected model, fitted to the series.
    fit_aic_ : float
        AIC of the selected model.

    References
    ----------
    .. [1] R. J. Hyndman and Y. Khandakar, Automatic time series forecasting: the
       forecast package for R. Journal of Statistical Software, 27(3), 2008.
    """

    _tags = {
        "capability:horizon": False,  # cannot fit to a horizon other than 1
    }

    def __init__(
        self,
        seasonal_period: int = 1,
        max_p: int = 3,
        max_d: int = 3,
        max_q: int = 2,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
    ):
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.P_ = 0
        self.D_ = 0
        self.Q_ = 0
        self.constant_term_ = False
        self.final_model_ = None
        super().__init__(horizon=1, axis=1)

    def _valid(self, order, data):
        """Whether ``(constant, p, q, P, Q)`` is within limits and can be fitted."""
        c, p, q, P, Q = order
        m = self.seasonal_period
        if min(p, q, P, Q) < 0:
            return False
        if p > self.max_p or q > self.max_q or P > self.max_P or Q > self.max_Q:
            return False
        if m == 1 and P + Q > 0:
            return False
        return len(data) > max(p + P * m, q + Q * m) + c + p + q + P + Q

    def _score(self, order, data, max_iter):
        """Fit a candidate, returning its parameters and AIC, inf if rejected."""
        c, p, q, P, Q = order
        model = _build_model(c == 1, p, q, P, Q, self.seasonal_period)
        params, aic = _fit_candidate(data, model, max_iter)
        if not _roots_clear_of_unit_circle(params, model):
            aic = np.inf
        return params, aic

    def _search(self, data, constant):
        """Stepwise search for the best ``(constant, p, q, P, Q)``."""
        best, best_params, best_aic = None, None, np.inf
        for start in [(0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1), (2, 2, 1, 1)]:
            order = (constant, *start)
            if self._valid(order, data):
                params, aic = self._score(order, data, 200)
                if aic < best_aic:
                    best, best_params, best_aic = order, params, aic
        if best is None:
            return None, None, np.inf

        # Neighbours: the seasonal or the non-seasonal orders move, as AutoARIMA's
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        steps = [(a, b, 0, 0) for a, b in moves]
        if self.seasonal_period > 1:
            steps += [(0, 0, a, b) for a, b in moves]
        # a repeated neighbour cannot improve, as the best AIC only falls
        tried = set()
        while True:
            # first improvement: move to the first better neighbour and search
            # again from there, as AutoARIMA does
            better_model = False
            centre = best
            for step in steps:
                orders = tuple(int(o + s) for o, s in zip(centre[1:], step))
                for constant_term in [0, 1]:
                    order = (constant_term, *orders)
                    if order in tried or not self._valid(order, data):
                        continue
                    tried.add(order)
                    params, aic = self._score(order, data, 500)
                    if aic < best_aic:
                        best, best_params, best_aic = order, params, aic
                        better_model = True
                        break
                if better_model:
                    break
            if not better_model:
                break
        return best, best_params, best_aic

    def _fit(self, y, exog=None):
        if self.seasonal_period < 1:
            raise ValueError("seasonal_period must be at least 1.")
        series = np.array(y.squeeze(), dtype=np.float64)
        m = self.seasonal_period
        self.d_, self.D_ = _select_differencing(series, m, self.max_d, self.max_D)
        data = _difference(series, _get_diff_poly(self.d_, self.D_, m))
        constant = 1 if self.d_ + self.D_ == 0 else 0
        best, parameters, self.fit_aic_ = self._search(data, constant)
        if best is None or not np.isfinite(self.fit_aic_):
            # Nothing could be fitted, for example a constant differenced series
            best, parameters = (0, 0, 0, 0, 0), np.empty(0)
        c, self.p_, self.q_, self.P_, self.Q_ = best
        self.constant_term_ = c == 1
        self.final_model_ = SARIMA(
            self.p_,
            self.d_,
            self.q_,
            self.P_,
            self.D_,
            self.Q_,
            seasonal_period=m,
            use_constant=self.constant_term_,
        )
        # Keep the parameters found in the search rather than refitting, so the
        # final model is the one that was scored and checked for near unit roots.
        # This mirrors BaseForecaster.fit (and its fit timer) for the final model:
        # keep in step if fit records anything else.
        start = time.perf_counter()
        self.final_model_._fit_with_parameters(y, parameters=parameters)
        self.final_model_.fit_time_millis_ = int((time.perf_counter() - start) * 1000)
        self.final_model_.metadata_ = dict(self.metadata_)
        self.final_model_.is_fitted = True
        self.forecast_ = self.final_model_.forecast_
        return self

    def _predict(self, y, exog=None):
        return self.final_model_.predict(y)

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
        return self.final_model_._iterative_forecast_from_fitted(prediction_horizon)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator."""
        return {"seasonal_period": 4, "max_p": 1, "max_q": 1, "max_P": 1, "max_Q": 1}
