"""4Theta forecaster: automatic selection among generalised Theta models."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["FourTheta"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._brent import NEXT, X, brent_start, brent_update
from aeon.forecasting.utils._seasonality import (
    _seasonal_decompose_numba,
    _seasonal_forecast,
    acf_seasonal_test,
)

# Two sided 90% normal quantile, as used by the authors.
_SEASONAL_QUANTILE = 1.645
_MULTIPLICATIVE_FLOOR = 1e-8
# Bounds on the SES smoothing parameter, those of R's ets.
_ALPHA_LOWER = 1e-4
_ALPHA_UPPER = 0.9999
_ALPHA_TOL = 1e-6
_MAX_ITER = 100
_INFEASIBLE = 1e300

_NONE, _MULT, _ADD = 0, 1, 2
_LINEAR, _EXPONENTIAL = 0, 1
_SEASONAL_NAMES = ("N", "M", "A")
_EXPRESSION_NAMES = ("", "M", "A")
_TREND_NAMES = ("Lrl", "Exp")


class FourTheta(BaseForecaster, IterativeForecastingMixin):
    r"""4Theta forecaster, automatic selection among generalised Theta models.

    Classic Theta [1]_ combines a linear trend line with simple exponential
    smoothing (SES) of a Theta line, a series with its curvature scaled by
    :math:`\theta`, using a fixed :math:`\theta = 2`. Spiliotis et al. [2]_
    generalise this in three ways: the trend may be exponential, :math:`\theta`
    is optimised, and the Theta line and the combination may be multiplicative.

    The series is scaled by its mean and tested for seasonality. Each candidate
    model is then fitted by choosing :math:`\theta` to minimise the in-sample
    one-step mean absolute error (MAE) using Brent's method, and the candidate
    with the lowest MAE is used to forecast.

    Candidates are named by seasonality (``N``, ``M`` or ``A``), expression
    (``M`` or ``A``) and trend (``Lrl`` or ``Exp``). The eight in the pool are
    ``N M Exp``, ``N A Lrl`` and ``N A Exp`` for a series that does not test as
    seasonal, and ``M M Exp``, ``M A Lrl``, ``M A Exp``, ``A A Lrl`` and
    ``A A Exp`` for one that does. Only additive linear models are candidates
    for a series with values that are not positive.

    Exogenous variables and prediction intervals are not supported.

    Parameters
    ----------
    seasonal_period : int, default=1
        Seasonal period. ``1`` means no seasonality. Seasonality is only tested
        for when there are at least three seasonal cycles.
    theta_bounds : tuple of float, default=(1.0, 3.0)
        Search interval for :math:`\theta`. The lower bound must be at least 1.
    tol : float, default=1.220703125e-4
        Brent tolerance on :math:`\theta`. The default is that of R's
        ``optimize``.

    Attributes
    ----------
    model_ : str
        Selected model, for example ``"M A Exp"``.
    theta_ : float
        Fitted :math:`\theta`.
    alpha_ : float
        Fitted SES smoothing parameter.
    initial_level_ : float
        Fitted SES initial level, on the scale of the mean scaled series.
    level_ : float
        Final SES level, on the scale of the mean scaled series.
    a_, b_ : float
        Intercept and slope of the trend against zero based time, fitted to the
        seasonally adjusted mean scaled series, or to its log if the trend is
        exponential.
    mae_ : float
        In-sample MAE of the selected model on the mean scaled series.
    seasonal_ : bool
        Whether the series tested as seasonal.
    seasonal_factors_ : np.ndarray or None
        Seasonal factors of length ``seasonal_period``, or ``None`` if the
        selected model is not seasonal.
    fitted_values_ : np.ndarray
        In-sample one-step fitted values.
    residuals_ : np.ndarray
        In-sample residuals, ``y - fitted_values_``.
    forecast_ : float
        One step ahead forecast from the series seen in ``fit``.

    Notes
    -----
    The candidates, seasonal test and selection follow the authors' ``AutoTheta``
    R code. It differs from that code in these ways.

    - SES is estimated by Brent's method on the smoothing parameter, with the
      initial level found in closed form, rather than by Nelder-Mead over both.
      The objective is the same SSE.
    - Decomposition and trend are estimated once for each candidate rather than
      for every evaluation of :math:`\theta`.
    - Forecasts are floored at zero only if all the training data are
      non-negative. R floors unconditionally, and on the mean scaled series,
      which reverses the floor for a series with a negative mean.
    - A candidate is skipped if it cannot be computed: ``A A Exp`` when the
      seasonally adjusted series is not positive, or ``M A Lrl`` when a
      multiplicative decomposition is not possible.
    - The seasonal test is always evaluated at ``seasonal_period``. In R it fails
      silently if ``seasonal_period > 10 log10(n)``.

    The name follows the authors' M4 submission [3]_ and avoids confusion with
    ``AutoTheta`` in statsforecast, which is a different algorithm. The M4 code
    has a pool of twelve models with fallbacks when a model is unstable.

    References
    ----------
    .. [1] Assimakopoulos, V. and Nikolopoulos, K. (2000). The theta model: a
       decomposition approach to forecasting. International Journal of
       Forecasting, 16(4), 521-530.
    .. [2] Spiliotis, E., Assimakopoulos, V. and Makridakis, S. (2020).
       Generalizing the Theta method for automatic forecasting. European Journal
       of Operational Research, 284(2), 550-558.
    .. [3] Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020). The M4
       Competition: 100,000 time series and 61 forecasting methods. International
       Journal of Forecasting, 36(1), 54-74.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting.stats import FourTheta
    >>> y = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.2, 4.9, 5.3, 5.6])
    >>> forecaster = FourTheta()
    >>> forecaster.iterative_forecast(y, prediction_horizon=3).shape
    (3,)
    >>> forecaster.model_ in ("N M Exp", "N A Lrl", "N A Exp")
    True
    """

    _tags = {"capability:horizon": False}

    def __init__(self, seasonal_period=1, theta_bounds=(1.0, 3.0), tol=1.220703125e-4):
        self.seasonal_period = seasonal_period
        self.theta_bounds = theta_bounds
        self.tol = tol
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        y = np.asarray(y, dtype=np.float64).squeeze()
        if y.shape[0] < 3:
            raise ValueError("FourTheta requires at least 3 observations.")
        m = self.seasonal_period
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("seasonal_period must be a positive integer.")
        lower, upper = self.theta_bounds
        if lower < 1.0 or lower >= upper:
            raise ValueError("theta_bounds must satisfy 1 <= lower < upper.")

        self._base = float(y.mean())
        if self._base == 0.0 or not np.isfinite(self._base):
            self._base = 1.0
        self._non_negative = bool(y.min() >= 0.0)
        self._n = y.shape[0]
        scaled = y / self._base

        self.seasonal_ = bool(
            m > 1
            and y.shape[0] >= 3 * m
            and acf_seasonal_test(scaled, int(m), _SEASONAL_QUANTILE)
        )
        (
            seasonal,
            expression,
            trend,
            self.theta_,
            self.alpha_,
            self.initial_level_,
            self.level_,
            self.a_,
            self.b_,
            self.mae_,
            factors,
            fitted,
        ) = _select_model(
            scaled,
            int(m),
            self.seasonal_,
            bool(y.min() > 0.0),
            float(lower),
            float(upper),
            float(self.tol),
        )
        self._seasonal_type = int(seasonal)
        self._expression = int(expression)
        self._trend = int(trend)
        self.model_ = " ".join(
            (
                _SEASONAL_NAMES[seasonal],
                _EXPRESSION_NAMES[expression],
                _TREND_NAMES[trend],
            )
        )
        self.seasonal_factors_ = factors if seasonal != _NONE else None
        self.fitted_values_ = fitted * self._base
        self.residuals_ = y - self.fitted_values_
        self.forecast_ = float(self._forecast_from_state(1)[0])
        return self

    def _forecast_from_state(self, h):
        """Forecast ``h`` steps from the fitted state, on the original scale."""
        forecast = _project(
            self._n,
            h,
            self._expression,
            self._trend,
            self.theta_,
            self.level_,
            self.a_,
            self.b_,
        )
        if self._seasonal_type != _NONE:
            future = _seasonal_forecast(
                self.seasonal_factors_, h, self.seasonal_period, self._n
            )
            if self._seasonal_type == _ADD:
                forecast = forecast + future
            else:
                forecast = forecast * future
        forecast = forecast * self._base
        if self._non_negative:
            forecast = np.maximum(forecast, 0.0)
        return forecast

    def _predict(self, y, exog=None):
        """Return the stored one step forecast from the series seen in ``fit``."""
        return self.forecast_

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit to ``y`` and forecast ``prediction_horizon`` steps in closed form.

        ``exog`` and ``future_exog`` are accepted for signature compatibility
        with :class:`~aeon.forecasting.base.IterativeForecastingMixin` but are
        not supported; passing either raises :class:`NotImplementedError`.
        """
        if isinstance(prediction_horizon, bool) or not isinstance(
            prediction_horizon, (int, np.integer)
        ):
            raise TypeError("prediction_horizon must be an integer.")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        if exog is not None or future_exog is not None:
            raise NotImplementedError("FourTheta does not support exog.")
        self.fit(y)
        return self._forecast_from_state(int(prediction_horizon))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"seasonal_period": 4}


@njit(cache=True, fastmath={"contract"})
def _ses_sse(z, alpha):
    """Return the SES SSE and initial level for a given smoothing parameter.

    With ``alpha`` fixed each one-step error is linear in the initial level
    ``l0``, ``e_t = r_t - (1 - alpha)^t l0``, where ``r_t`` is the error from
    starting at zero, so the best ``l0`` is a least squares estimate in closed
    form.
    """
    beta = 1.0 - alpha
    level = 0.0
    power = 1.0
    srr = 0.0
    srp = 0.0
    spp = 0.0
    for t in range(z.shape[0]):
        r = z[t] - level
        srr += r * r
        srp += r * power
        spp += power * power
        level += alpha * r
        power *= beta
    l0 = srp / spp
    return srr - l0 * srp, l0


@njit(cache=True, fastmath={"contract"})
def _fit_ses(z):
    """Fit SES by minimising SSE, returning ``alpha`` and the initial level."""
    state = brent_start(_ALPHA_LOWER, _ALPHA_UPPER)
    for _ in range(_MAX_ITER):
        sse, _ = _ses_sse(z, state[NEXT])
        if not np.isfinite(sse):
            sse = _INFEASIBLE
        if brent_update(state, sse, _ALPHA_TOL):
            break
    alpha = state[X]
    _, l0 = _ses_sse(z, alpha)
    return alpha, l0


@njit(cache=True, fastmath={"contract"})
def _theta_line(deseasonalised, log_deseasonalised, trend, log_trend, theta, expr, z):
    """Write the Theta line into ``z``."""
    if expr == _ADD:
        for t in range(z.shape[0]):
            z[t] = theta * deseasonalised[t] + (1.0 - theta) * trend[t]
    else:
        for t in range(z.shape[0]):
            z[t] = np.exp(theta * log_deseasonalised[t] + (1.0 - theta) * log_trend[t])


@njit(cache=True, fastmath={"contract"})
def _combine(level, trend_value, log_trend_value, theta, expr):
    """Combine an SES level with the trend line."""
    if expr == _ADD:
        return level / theta + (1.0 - 1.0 / theta) * trend_value
    if level <= 0.0:
        return np.nan
    return np.exp(np.log(level) / theta + (1.0 - 1.0 / theta) * log_trend_value)


@njit(cache=True, fastmath={"contract"})
def _in_sample(
    y,
    deseasonalised,
    log_deseasonalised,
    trend,
    log_trend,
    factors,
    seasonal,
    expr,
    theta,
    z,
    fitted,
):
    """Fit SES to the Theta line and write fitted values, returning its state."""
    n = y.shape[0]
    m = factors.shape[0]
    _theta_line(deseasonalised, log_deseasonalised, trend, log_trend, theta, expr, z)
    alpha, l0 = _fit_ses(z)
    level = l0
    total = 0.0
    for t in range(n):
        f = _combine(level, trend[t], log_trend[t], theta, expr)
        if seasonal == _ADD:
            f += factors[t % m]
        elif seasonal == _MULT:
            f *= factors[t % m]
        fitted[t] = f
        total += abs(y[t] - f)
        level += alpha * (z[t] - level)
    return total / n, alpha, l0, level


@njit(cache=True, fastmath={"contract"})
def _fit_candidate(
    y,
    deseasonalised,
    factors,
    seasonal,
    expr,
    trend_type,
    lower,
    upper,
    tol,
):
    """Choose theta for one candidate, returning its parameters and fitted values."""
    n = y.shape[0]
    log_deseasonalised = np.empty(n, dtype=np.float64)
    target = deseasonalised
    if trend_type == _EXPONENTIAL or expr == _MULT:
        for t in range(n):
            if deseasonalised[t] <= 0.0:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, y
            log_deseasonalised[t] = np.log(deseasonalised[t])
        if trend_type == _EXPONENTIAL:
            target = log_deseasonalised

    a, b = _ols_on_time(target)
    trend = np.empty(n, dtype=np.float64)
    log_trend = np.empty(n, dtype=np.float64)
    for t in range(n):
        line = a + b * t
        if trend_type == _EXPONENTIAL:
            log_trend[t] = line
            trend[t] = np.exp(line)
        else:
            trend[t] = line
            log_trend[t] = 0.0

    z = np.empty(n, dtype=np.float64)
    fitted = np.empty(n, dtype=np.float64)
    state = brent_start(lower, upper)
    for _ in range(_MAX_ITER):
        mae, _, _, _ = _in_sample(
            y,
            deseasonalised,
            log_deseasonalised,
            trend,
            log_trend,
            factors,
            seasonal,
            expr,
            state[NEXT],
            z,
            fitted,
        )
        if not np.isfinite(mae):
            mae = _INFEASIBLE
        if brent_update(state, mae, tol):
            break
    theta = state[X]
    mae, alpha, l0, level = _in_sample(
        y,
        deseasonalised,
        log_deseasonalised,
        trend,
        log_trend,
        factors,
        seasonal,
        expr,
        theta,
        z,
        fitted,
    )
    return np.isfinite(mae), mae, theta, alpha, l0, level, a, b, fitted


@njit(cache=True, fastmath={"contract"})
def _ols_on_time(x):
    """Return intercept and slope of ``x`` regressed on zero based time."""
    n = x.shape[0]
    t_mean = (n - 1) / 2.0
    x_mean = 0.0
    for t in range(n):
        x_mean += x[t]
    x_mean /= n
    sxy = 0.0
    sxx = 0.0
    for t in range(n):
        dt = t - t_mean
        sxy += dt * (x[t] - x_mean)
        sxx += dt * dt
    b = sxy / sxx
    return x_mean - b * t_mean, b


@njit(cache=True)
def _select_model(y, m, is_seasonal, positive, lower, upper, tol):
    """Fit every candidate for a mean scaled series and return the best."""
    seasonal_types = [_MULT, _ADD] if is_seasonal else [_NONE]

    # (seasonal, expression, trend) in the order of the authors' code, so that
    # exact ties are resolved the same way.
    candidates = []
    for s in seasonal_types:
        if positive:
            if s != _ADD:
                candidates.append((s, _MULT, _EXPONENTIAL))
            candidates.append((s, _ADD, _LINEAR))
            candidates.append((s, _ADD, _EXPONENTIAL))
        else:
            candidates.append((s, _ADD, _LINEAR))

    identity = np.zeros(1, dtype=np.float64)
    best_mae = np.inf
    best = (_NONE, _ADD, _LINEAR, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0)
    best_factors = identity
    best_fitted = y.copy()

    mult_factors = identity
    mult_adjusted = y
    mult_ok = False
    add_factors = identity
    add_adjusted = y
    add_ok = False
    if is_seasonal:
        code, mult_factors, mult_adjusted = _seasonal_decompose_numba(
            y, m, True, _MULTIPLICATIVE_FLOOR
        )
        mult_ok = code == 2
        code, add_factors, add_adjusted = _seasonal_decompose_numba(
            y, m, False, _MULTIPLICATIVE_FLOOR
        )
        add_ok = code == 1

    for s, expr, trend_type in candidates:
        if s == _MULT:
            if not mult_ok:
                continue
            factors, adjusted = mult_factors, mult_adjusted
        elif s == _ADD:
            if not add_ok:
                continue
            factors, adjusted = add_factors, add_adjusted
        else:
            factors, adjusted = identity, y
        ok, mae, theta, alpha, l0, level, a, b, fitted = _fit_candidate(
            y, adjusted, factors, s, expr, trend_type, lower, upper, tol
        )
        if ok and mae < best_mae:
            best_mae = mae
            best = (s, expr, trend_type, theta, alpha, l0, level, a, b)
            best_factors = factors
            best_fitted = fitted.copy()

    s, expr, trend_type, theta, alpha, l0, level, a, b = best
    return (
        s,
        expr,
        trend_type,
        theta,
        alpha,
        l0,
        level,
        a,
        b,
        best_mae,
        best_factors,
        best_fitted,
    )


@njit(cache=True, fastmath={"contract"})
def _project(n, h, expr, trend_type, theta, level, a, b):
    """Extrapolate the trend and combine with the flat SES level."""
    forecast = np.empty(h, dtype=np.float64)
    for k in range(h):
        line = a + b * (n + k)
        if trend_type == _EXPONENTIAL:
            forecast[k] = _combine(level, np.exp(line), line, theta, expr)
        else:
            forecast[k] = _combine(level, line, 0.0, theta, expr)
    return forecast
