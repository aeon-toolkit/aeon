"""Theta forecaster."""

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster


class Theta(BaseForecaster):
    """Theta forecaster.

    Parameters
    ----------
    theta : float, default=2.0
        The theta parameter. Classical Theta method uses theta=2.0.
    """

    _tags = {"capability:horizon": False, "fit_is_empty": True}

    def __init__(self, theta=2.0):
        self.theta = theta
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        return self

    def _predict(self, y, exog=None):
        y = y.squeeze()
        f = _fit_predict_numba(y, 1, self.theta)
        return f[0]

    def iterative_forecast(self, y, prediction_horizon):
        y = y.squeeze()
        f = _fit_predict_numba(y, prediction_horizon, self.theta)
        return f


@njit(cache=True, fastmath=True)
def _fit_predict_numba(y: np.ndarray, h: int, theta: float = 2.0) -> np.ndarray:
    n = len(y)
    t = np.arange(n)

    # Step 1: Linear regression (least squares) for trend
    t_mean = t.mean()
    y_mean = y.mean()
    cov_ty = np.dot(t - t_mean, y - y_mean)
    var_t = np.dot(t - t_mean, t - t_mean)
    b = cov_ty / var_t
    a = y_mean - b * t_mean

    # Extend trend to n + h steps
    trend = a + b * np.arange(n + h)

    # Step 2: Theta line: θ * y + (1 - θ) * trend
    theta_line = theta * y + (1 - theta) * trend[:n]

    # Step 3: SES forecast on theta line (constant = last value)
    ses_level = theta_line[-1]

    # Step 4: Combine SES forecast with trend
    forecast = np.empty(h)
    for i in range(h):
        trend_part = trend[n + i]
        forecast[i] = (theta / 2) * ses_level + (1 - theta / 2) * trend_part

    return forecast
