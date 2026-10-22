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
