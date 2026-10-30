"""ARIMA Utility Function."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _extract_arma_params(params, model):
    """Extract ARIMA parameters from the parameter vector."""
    n_parts = len(model)
    starts = np.zeros(n_parts, dtype=np.int32)
    for i in range(1, n_parts):
        starts[i] = starts[i - 1] + model[i - 1]

    max_len = np.max(model)
    result = np.empty((n_parts, max_len), dtype=params.dtype)
    for i in range(n_parts):
        for j in range(max_len):
            result[i, j] = np.nan

    for i in range(n_parts):
        length = model[i]
        start = starts[i]
        for j in range(length):
            result[i, j] = params[start + j]

    return result


@njit(cache=True, fastmath={"contract"})
def _extract_sarima_params(params, model):
    """Unpack ``[c?, phi..., theta..., Phi..., Theta...]`` into its five parts.

    ``model`` is ``[constant, p, q, P, Q, m, n_ar_lags, n_ma_lags, ar_lags...,
    ma_lags...]``.
    """
    p, q, P, Q = model[1], model[2], model[3], model[4]
    c = params[0] if model[0] else 0.0
    idx = 1 if model[0] else 0
    phi = params[idx : idx + p]
    idx += p
    theta = params[idx : idx + q]
    idx += q
    Phi = params[idx : idx + P]
    idx += P
    Theta = params[idx : idx + Q]
    return c, phi, theta, Phi, Theta


@njit(cache=True, fastmath={"contract"})
def _sarima_lags(model):
    """Return the sparse AR lags, MA lags and first residual time of ``model``."""
    n_ar_lags, n_ma_lags = model[6], model[7]
    ar_lags = model[8 : 8 + n_ar_lags]
    ma_lags = model[8 + n_ar_lags : 8 + n_ar_lags + n_ma_lags]
    start = 0
    if n_ar_lags > 0:
        start = max(start, ar_lags[-1])
    if n_ma_lags > 0:
        start = max(start, ma_lags[-1])
    return ar_lags, ma_lags, start


@njit(cache=True, fastmath={"contract"})
def _seasonal_polynomial(coefs, seasonal_coefs, m):
    """Expand ``(1 + sum(coefs B^i))(1 + sum(seasonal_coefs B^(i m)))``."""
    poly = np.zeros(len(coefs) + len(seasonal_coefs) * m + 1)
    for j in range(len(seasonal_coefs) + 1):
        scale = 1.0 if j == 0 else seasonal_coefs[j - 1]
        poly[j * m] += scale
        for i in range(len(coefs)):
            poly[i + 1 + j * m] += scale * coefs[i]
    return poly


@njit(cache=True, fastmath={"contract"})
def _sarima_coefficients(params, model):
    """Return the constant and the AR and MA coefficients at the sparse lags."""
    c, phi, theta, Phi, Theta = _extract_sarima_params(params, model)
    m = model[5]
    ar_lags, ma_lags, _ = _sarima_lags(model)
    ar_poly = _seasonal_polynomial(-phi, -Phi, m)
    ma_poly = _seasonal_polynomial(theta, Theta, m)
    ar_values = np.empty(len(ar_lags))
    for i in range(len(ar_lags)):
        ar_values[i] = -ar_poly[ar_lags[i]]
    ma_values = np.empty(len(ma_lags))
    for i in range(len(ma_lags)):
        ma_values[i] = ma_poly[ma_lags[i]]
    return c, ar_values, ma_values


@njit(fastmath=True, cache=True)
def _ets_is_damped(model):
    """Whether a trend in ``model`` is damped (phi estimated).

    ``model`` is ``[error, trend, seasonality, period]`` with an optional fifth
    element, 1 for a damped trend and 0 for an undamped one. Four element models
    are damped, the behaviour before the flag existed.
    """
    return len(model) < 5 or model[4] != 0


@njit(fastmath=True, cache=True)
def _ets_num_params(model):
    """Return the number of smoothing parameters estimated for ``model``."""
    n = 1
    if model[1] != 0:
        n += 2 if _ets_is_damped(model) else 1
    if model[2] != 0:
        n += 1
    return n


@njit(fastmath=True, cache=True)
def _extract_ets_params(params, model):
    """Unpack ``[alpha, (beta, (phi)), (gamma)]`` into alpha, beta, gamma, phi."""
    trend_type = model[1]
    seasonality_type = model[2]
    alpha = params[0]
    beta = 0.0
    phi = 1.0
    gamma = 0.0
    i = 1
    if trend_type != 0:
        beta = params[i]
        i += 1
        if _ets_is_damped(model):
            phi = params[i]
            i += 1
    if seasonality_type != 0:
        gamma = params[i]
    return alpha, beta, gamma, phi
