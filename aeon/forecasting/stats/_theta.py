import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class Theta(BaseForecaster, IterativeForecastingMixin):
    r"""Classical Theta forecaster.

    Overview
    --------
    This is a non-seasonal implementation of the *classical* Theta method
    (M3 competition winner). It decomposes a univariate series into:
    (i) a linear trend component (the "θ=0" line) estimated by OLS on time,
    and (ii) a curvature/level component obtained by applying Simple
    Exponential Smoothing (SES) to the "theta line"
    :math:`\\theta y_t + (1-\\theta) \\hat{y}^{\\text{trend}}_t`, with
    :math:`\\theta=2` by default. Multi-step forecasts are formed by an equal-
    weight (or user-specified) convex combination of the extrapolated trend and
    the flat SES level, yielding closed-form forecasts for any horizon.

    Parameters
    ----------
    theta : float, default=2.0
        Theta parameter used to form the theta line. ``theta=2`` reproduces the
        classical method; values >1 accentuate curvature, <1 dampen it.
    weight : float, default=0.5
        Weight assigned to the trend component in the final combination.
        The SES component receives ``(1 - weight)``. Values are clipped to
        ``[0, 1]``.

    Attributes
    ----------
    a_ : float
        Estimated trend intercept from OLS on time.
    b_ : float
        Estimated trend slope from OLS on time.
    alpha_ : float
        SES smoothing parameter selected by in-sample SSE minimisation on the
        theta line.
    forecast_ : float
        Stored one-step-ahead forecast computed at ``fit`` time.
    _tags : dict
        Includes ``{"capability:horizon": False}`` to indicate closed-form
        multi-step forecasting (no direct multi-output training).

    Notes
    -----
    - The classical Theta model is equivalent to an ETS(A,N,N) with drift under
      certain conditions; see Hyndman & Billah (2003) for details. This
      implementation follows the classical two-theta construction with
      :math:`\\theta=2` and an SES fit on the theta line.
    - This is a *local* method: parameters are estimated per series; the model
      is not intended to generalise to unseen series without re-fitting.

    References
    ----------
    .. [1] Assimakopoulos, V. and Nikolopoulos, K. (2000).
           "The Theta model: a decomposition approach to forecasting."
           *International Journal of Forecasting*, 16(4), 521–530.
    .. [2] Hyndman, R.J. and Billah, B. (2003).
           "Unmasking the Theta method."
           *International Journal of Forecasting*, 19(2), 287–290.
    """

    _tags = {"capability:horizon": False}

    def __init__(self, theta=2.0, weight=0.5):
        self.theta = float(theta)
        self.weight = float(weight)
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        if y.shape[1] < 3:
            raise ValueError(
                "Theta forecaster requires at least 3 observations in the series."
            )
        y = np.asarray(y).squeeze().astype(np.float64)
        f, self.a_, self.b_, self.alpha_ = _fit_predict_numba(
            y, 1, self.theta, self.weight
        )
        self.forecast_ = f[-1]
        return self

    def _predict(self, y, exog=None):
        return self.forecast_

    def iterative_forecast(self, y, prediction_horizon):
        y = np.asarray(y).squeeze().astype(np.float64)
        f, _, _, _ = _fit_predict_numba(y, prediction_horizon, self.theta, self.weight)
        return f


@njit(cache=True, fastmath=True)
def _fit_predict_numba(y: np.ndarray, h: int, theta: float, weight: float):
    n = len(y)

    # --- Trend component (theta=0)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    y_mean = y.mean()
    dt = t - t_mean
    dy = y - y_mean
    var_t = (dt * dt).sum()

    if var_t <= 1e-12:
        b = 0.0
        a = y_mean
    else:
        b = (dt * dy).sum() / var_t
        a = y_mean - b * t_mean

    trend_in = a + b * t
    trend_fut = a + b * np.arange(n, n + h, dtype=np.float64)

    # --- Theta line (classical uses theta=2)
    theta_line = theta * y + (1.0 - theta) * trend_in

    # --- Estimate alpha for SES by SSE minimisation
    best_sse = 1e300
    alpha = 0.1
    for k in range(1, 101):
        a_try = k / 100.0
        sse = _ses_sse(theta_line, a_try)
        if sse < best_sse:
            best_sse = sse
            alpha = a_try

    # --- SES forecast
    level = _ses_last_level(theta_line, alpha)
    ses_fut = np.full(h, level)

    # --- Combine components
    w = min(max(weight, 0.0), 1.0)
    forecast = w * trend_fut + (1.0 - w) * ses_fut

    return forecast, a, b, alpha


@njit(cache=True, fastmath=True)
def _ses_sse(y: np.ndarray, alpha: float) -> float:
    level = y[0]
    sse = 0.0
    for t in range(1, y.shape[0]):
        e = y[t] - level
        sse += e * e
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse


@njit(cache=True, fastmath=True)
def _ses_last_level(y: np.ndarray, alpha: float) -> float:
    level = y[0]
    for t in range(1, y.shape[0]):
        level = alpha * y[t] + (1.0 - alpha) * level
    return level
