"""Loss functions for optimiser."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _arima_fit(params, data, model):
    """Calculate the AIC of an ARIMA model given the parameters."""
    formatted_params = _extract_params(params, model)  # Extract parameters

    # Initialize residuals
    n = len(data)
    residuals = np.zeros(n)
    c = formatted_params[0][0] if model[0] else 0
    p = model[1]
    q = model[2]
    for t in range(n):
        ar_term = 0
        max_ar = min(p, t)
        for j in range(max_ar):
            ar_term += formatted_params[1, j] * data[t - j - 1]
        ma_term = 0
        max_ma = min(q, t)
        for j in range(max_ma):
            ma_term += formatted_params[2, j] * residuals[t - j - 1]
        y_hat = c + ar_term + ma_term
        residuals[t] = data[t] - y_hat
    variance = np.mean(residuals**2)
    LOG_2PI = 1.8378770664093453
    likelihood = n * (LOG_2PI + np.log(variance) + 1.0)
    return likelihood + 2 * len(params)


@njit(cache=True, fastmath=True)
def _extract_params(params, model):
    """Extract ARIMA parameters from the parameter vector."""
    starts = np.cumsum(np.concatenate((np.zeros(1, dtype=np.int32), model[:-1])))
    n = len(starts)
    max_len = np.max(model)
    result = np.full((n, max_len), np.nan, dtype=params.dtype)
    for i in range(n):
        length = model[i]
        start = starts[i]
        result[i, :length] = params[start : start + length]
    return result
