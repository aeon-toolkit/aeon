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
    fitted_values = np.zeros(n)
    c = formatted_params[0][0] if model[0] else 0
    p = model[1]
    q = model[2]
    # AR part
    phi = formatted_params[1][:p]
    theta = formatted_params[2][:q]
    for t in range(n):
        ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])
        ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])
        fitted_values[t] = c + ar_term + ma_term
        residuals[t] = data[t] - fitted_values[t]
    return _aic(residuals, len(params))


@njit(cache=True, fastmath=True)
def _in_sample_forecast(
    data,
    model,
    t,
    formatted_params,
    residuals,
    p,
    q,
    phi,
):
    """Calculate the ARMA forecast for time t."""
    p = model[1]
    q = model[2]
    # AR part
    phi = formatted_params[1][:p]
    ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])

    # MA part
    theta = formatted_params[2][:q]
    ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])

    c = formatted_params[0][0] if model[0] else 0
    y_hat = c + ar_term + ma_term
    return y_hat


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


@njit(cache=True, fastmath=True)
def _aic(residuals, num_params):
    """Calculate the log-likelihood of a model."""
    variance = np.mean(residuals**2)
    likelihood = len(residuals) * (np.log(2 * np.pi) + np.log(variance) + 1)
    return likelihood + 2 * num_params
