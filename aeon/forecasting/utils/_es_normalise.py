"""Exponential smoothing (ES) level and seasonal normalisation.

Tools to normalise a series by the level and seasonal components of a
multiplicative exponential smoothing model with no trend (ETS(M,N,M), or ETS(M,N,N)
when there is no seasonality), as used to preprocess the inputs to the RNN in the
ES-RNN method that won the M4 competition [1]_.

Each window ending at time ``t`` is divided by the ES level at ``t`` and by each
point's own seasonal factor, so a single scale-free representation is produced for
every window. Targets ``h`` steps ahead are divided by the same level and the most
recent seasonal factor for their phase that is known at time ``t``, so no future
information leaks into the normalisation.

References
----------
.. [1] Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent
   neural networks for time series forecasting. International Journal of
   Forecasting, 36(1), 75-85.
"""

__maintainer__ = ["AlexBanwell"]
__all__ = [
    "fit_es_components",
    "es_components",
    "es_seasonal_index",
    "es_normalise_windows",
    "es_denormalise",
]

import numpy as np
from numba import njit


def fit_es_components(y, seasonal_period=1, alpha=None, gamma=None, iterations=200):
    """Fit ES smoothing parameters and return the level and seasonal paths.

    Fits a multiplicative error, no trend ETS model with multiplicative seasonality
    (or no seasonality if ``seasonal_period`` is 1 or the series is shorter than two
    full seasons) using :class:`aeon.forecasting.stats.ETS`, then replays the state
    equations to record the level and seasonal factor at every time point.

    Parameters
    ----------
    y : np.ndarray
        1D strictly positive time series.
    seasonal_period : int, default=1
        Number of time points in a seasonal cycle.
    alpha : float or None, default=None
        Level smoothing parameter. If None it is estimated from ``y``.
    gamma : float or None, default=None
        Seasonal smoothing parameter. If None it is estimated from ``y``. Ignored if
        there is no seasonality.
    iterations : int, default=200
        Maximum number of optimiser iterations used when estimating parameters.

    Returns
    -------
    level : np.ndarray
        Shape ``(n_timepoints,)``. ``level[t]`` is the ES level after observing
        ``y[t]``.
    season : np.ndarray
        Shape ``(n_timepoints + m,)`` where ``m`` is the seasonal period used.
        ``season[t]`` is the seasonal factor applied to time ``t``; the final ``m``
        values are the factors for the ``m`` steps after the series.
    alpha : float
        Level smoothing parameter used.
    gamma : float
        Seasonal smoothing parameter used (0 if there is no seasonality).
    seasonal_period : int
        Seasonal period used, 1 if seasonality was dropped.
    """
    y = _check_series(y)
    m = _effective_period(len(y), seasonal_period)
    if alpha is None or (gamma is None and m > 1):
        from aeon.forecasting.stats import ETS

        ets = ETS(
            error_type="multiplicative",
            trend_type=None,
            seasonality_type="multiplicative" if m > 1 else None,
            seasonal_period=m,
            iterations=iterations,
        )
        ets.fit(y)
        if alpha is None:
            alpha = ets.alpha_
        if gamma is None:
            gamma = ets.gamma_
    if gamma is None:
        gamma = 0.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    gamma = float(np.clip(gamma, 0.0, 1.0)) if m > 1 else 0.0
    level, season = es_components(y, m, alpha, gamma)
    return level, season, alpha, gamma, m


def es_components(y, seasonal_period, alpha, gamma):
    """Return the ES level and seasonal paths for fixed smoothing parameters.

    Parameters
    ----------
    y : np.ndarray
        1D strictly positive time series.
    seasonal_period : int
        Number of time points in a seasonal cycle. Seasonality is dropped if the
        series is shorter than two full seasons.
    alpha : float
        Level smoothing parameter in [0, 1].
    gamma : float
        Seasonal smoothing parameter in [0, 1].

    Returns
    -------
    level : np.ndarray
        Shape ``(n_timepoints,)``, the level after observing each point.
    season : np.ndarray
        Shape ``(n_timepoints + m,)``, the seasonal factor for each time point and
        the ``m`` steps after the series.
    """
    y = _check_series(y)
    m = _effective_period(len(y), seasonal_period)
    return _es_states(y, m, float(alpha), float(gamma))


def es_seasonal_index(target, origin, n_season, seasonal_period):
    """Index of the seasonal factor for ``target`` known at time ``origin``.

    The factor for time ``tau`` becomes known after observing ``tau - m``, so at
    origin ``t`` factors up to ``t + m`` are available. Later targets reuse the most
    recent factor for the same phase.

    Parameters
    ----------
    target : int or np.ndarray
        Time index (or indices) of the value being normalised.
    origin : int or np.ndarray
        Time index of the last observation used, i.e. the end of the window.
    n_season : int
        Length of the seasonal path returned by :func:`es_components`.
    seasonal_period : int
        Seasonal period used to build the seasonal path.

    Returns
    -------
    int or np.ndarray
        Index into the seasonal path.
    """
    m = seasonal_period
    limit = np.minimum(origin + m, n_season - 1)
    over = np.maximum(target - limit, 0)
    return target - m * ((over + m - 1) // m)


def es_normalise_windows(y, level, season, window, horizon, seasonal_period, log=False):
    """Form ES normalised sliding windows and ``horizon`` ahead targets.

    Window ``i`` covers ``y[i : i + window]`` and ends at origin
    ``t = i + window - 1``. Each value ``y[j]`` in it is divided by
    ``level[t] * season[j]``; the target ``y[t + horizon]`` is divided by
    ``level[t]`` and the most recent seasonal factor for its phase known at ``t``.

    Parameters
    ----------
    y : np.ndarray
        1D strictly positive time series.
    level, season : np.ndarray
        ES level and seasonal paths from :func:`es_components` or
        :func:`fit_es_components`.
    window : int
        Number of points in each input window.
    horizon : int
        Number of steps ahead of the window end to take the target from.
    seasonal_period : int
        Seasonal period used to build ``season``.
    log : bool, default=False
        If True, take the natural log of the normalised values.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_windows, window)``, the normalised windows.
    z : np.ndarray
        Shape ``(n_windows,)``, the normalised targets.
    """
    y = _check_series(y)
    origins = np.arange(window - 1, len(y) - horizon)
    X = np.lib.stride_tricks.sliding_window_view(y, window)[: len(origins)]
    S = np.lib.stride_tricks.sliding_window_view(season[: len(y)], window)
    S = S[: len(origins)]
    X = X / (level[origins][:, None] * S)
    targets = origins + horizon
    s_idx = es_seasonal_index(targets, origins, len(season), seasonal_period)
    z = y[targets] / (level[origins] * season[s_idx])
    if log:
        X, z = np.log(X), np.log(z)
    return X, z


def es_denormalise(z, level, season, origin, horizon, seasonal_period, log=False):
    """Map a normalised ``horizon`` ahead prediction back to the original scale.

    Inverse of the target transform in :func:`es_normalise_windows`.

    Parameters
    ----------
    z : float or np.ndarray
        Normalised prediction(s).
    level, season : np.ndarray
        ES level and seasonal paths used for normalisation.
    origin : int
        Time index of the last observation in the window, usually ``len(y) - 1``.
    horizon : int
        Number of steps ahead of ``origin`` being predicted.
    seasonal_period : int
        Seasonal period used to build ``season``.
    log : bool, default=False
        Whether the normalised values were log transformed.

    Returns
    -------
    float or np.ndarray
        Prediction(s) on the original scale.
    """
    if log:
        z = np.exp(z)
    s_idx = es_seasonal_index(origin + horizon, origin, len(season), seasonal_period)
    return z * level[origin] * season[s_idx]


def _check_series(y):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) < 2:
        raise ValueError("ES normalisation requires at least two observations.")
    if not np.all(np.isfinite(y)):
        raise ValueError("ES normalisation requires finite observations.")
    if np.any(y <= 0):
        raise ValueError(
            "ES normalisation uses a multiplicative model and requires a strictly "
            "positive series."
        )
    return y


def _effective_period(n_timepoints, seasonal_period):
    m = int(seasonal_period)
    if m < 1:
        raise ValueError("seasonal_period must be a positive integer.")
    if n_timepoints < 2 * m:
        return 1
    return m


@njit(cache=True, fastmath=True)
def _es_states(y, m, alpha, gamma):
    """Run ETS(M,N,M) state equations, matching aeon's ETS initialisation."""
    n = len(y)
    level = np.empty(n)
    season = np.ones(n + m)
    init_level = np.mean(y[:m])
    if m > 1:
        # The first season initialises the states and is applied unchanged to the
        # second season, which is where the state updates start.
        for i in range(m):
            season[i] = y[i] / init_level
            season[i + m] = season[i]
    for i in range(m):
        level[i] = init_level
    lev = init_level
    for t in range(m, n):
        s = season[t]
        error = y[t] / (lev * s) - 1.0
        lev = lev * (1.0 + alpha * error)
        if m > 1:
            season[t + m] = s * (1.0 + gamma * error)
        level[t] = lev
    return level, season
