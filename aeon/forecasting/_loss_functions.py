"""Loss functions for optimiser."""

import numpy as np
from numba import njit

from aeon.forecasting._extract_paras import _extract_arma_params

LOG_2PI = 1.8378770664093453


@njit(cache=True, fastmath=True)
def _arima_fit(params, data, model):
    """Calculate the AIC of an ARIMA model given the parameters."""
    formatted_params = _extract_arma_params(params, model)  # Extract parameters

    # Initialize residuals
    n = len(data)
    residuals = np.zeros(n)
    c = formatted_params[0][0] if model[0] else 0
    p = model[1]
    q = model[2]
    for t in range(n):
        ar_term = 0.0
        max_ar = min(p, t)
        for j in range(max_ar):
            ar_term += formatted_params[1, j] * data[t - j - 1]
        ma_term = 0.0
        max_ma = min(q, t)
        for j in range(max_ma):
            ma_term += formatted_params[2, j] * residuals[t - j - 1]
        y_hat = c + ar_term + ma_term
        residuals[t] = data[t] - y_hat
    sse = 0.0
    for i in range(n):
        sse += residuals[i] ** 2
    variance = sse / n
    likelihood = n * (LOG_2PI + np.log(variance) + 1.0)
    k = len(params)
    return likelihood + 2 * k
