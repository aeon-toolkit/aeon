"""Theta forecaster."""

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster


class Theta(BaseForecaster):
    """Theta forecaster.

    The Theta forecaster [1] is a univariate forecasting method that combines
    linear extrapolation of the long-term trend with simple exponential smoothing
    of a transformed time series known as the "theta line". It was one of the
    best-performing methods in the M3 competition.

    It works by:
    - Fit a linear regression on the series against time to form a trend line.
    - Form a theta line through combining the original series and the trend line with a
    parameter theta.
    - Forecast the theta line with simple exponential smoothing.

    Theta is not a weight but a control parameter.

        theta_line = theta * y + (1 - theta) * trend

    A theta value of 0 gives the linear trend, theta of 1 gives the original series
    and values greater than one extenuates differences from the trend.  This
    implementation uses the classical two-theta approach with theta=2 by default,
    as proposed in [1].

    Parameters
    ----------
    theta : float, default=2.0
        The theta parameter. Classical Theta method uses theta=2.0.
        Values greater than 1 amplify the curvature; values less than 1 dampen it.

    Reference
    ---------
    .. [1] Assimakopoulos, V. and Nikolopoulos, K. (2000).
       "The Theta model: a decomposition approach to forecasting".
       *International Journal of Forecasting*, 16(4), 521–530.
    """

    _tags = {"capability:horizon": False}

    def __init__(self, theta=2.0, weight=0.5):
        self.theta = theta
        self.weight = weight
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        y = y.squeeze()
        self.forecast_ = _fit_predict_numba(y, 1, self.theta)[-1]
        return self

    def _predict(self, y, exog=None):
        return self.forecast_

    def iterative_forecast(self, y, prediction_horizon):
        y = y.squeeze()
        f = _fit_predict_numba(y, prediction_horizon, self.theta)
        return f


@njit(cache=True, fastmath=True)
def _fit_predict_numba(y: np.ndarray, h: int, theta: float = 2.0) -> np.ndarray:
    n = len(y)
    t = np.arange(n)

    # Step 1: Linear regression for trend (θ = 0)
    t_mean = t.mean()
    y_mean = y.mean()
    cov_ty = np.dot(t - t_mean, y - y_mean)
    var_t = np.dot(t - t_mean, t - t_mean)
    b = cov_ty / var_t
    a = y_mean - b * t_mean

    trend_in_sample = a + b * t  # θ = 0 trend (in-sample)
    trend_forecast = a + b * np.arange(n, n + h)  # future trend

    # Step 2: Theta=2 line
    theta_line = theta * y + (1 - theta) * trend_in_sample

    # Step 3: SES forecast = last value of theta=2 line
    ses_forecast = np.full(h, theta_line[-1])

    # Step 4: Combine
    forecast = 0.5 * trend_forecast + 0.5 * ses_forecast

    return forecast
