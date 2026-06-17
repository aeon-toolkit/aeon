"""Dynamic Optimised Theta Model (DOTM) forecaster."""

__maintainer__ = []
__all__ = ["DOTM"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class DOTM(BaseForecaster, IterativeForecastingMixin):
    """Dynamic Optimised Theta Model (DOTM) forecaster.

    Non-seasonal implementation of the Dynamic Optimised Theta Model (DOTM)
    proposed by Fiorucci et al. (2016). DOTM is a local univariate forecaster
    that estimates an initial level, smoothing parameter, and theta parameter
    unless they are fixed by the user. Seasonal decomposition, exogenous
    variables, and prediction intervals are not included in this first version.

    Parameters
    ----------
    initial_level : float or None, default=None
        Fixed initial level, ``ell0``. If ``None``, it is estimated.
    alpha : float or None, default=None
        Fixed smoothing parameter. If ``None``, it is estimated.
    theta : float or None, default=None
        Fixed theta parameter. If ``None``, it is estimated.
    initial_level_bounds : tuple of float, default=(-1e10, 1e10)
        Bounds for estimated ``initial_level``.
    alpha_bounds : tuple of float, default=(0.1, 0.99)
        Bounds for estimated ``alpha``.
    theta_bounds : tuple of float, default=(1.0, 1e10)
        Bounds for estimated ``theta``.
    max_iter : int, default=500
        Maximum number of Nelder-Mead iterations.
    tol : float, default=1e-6
        Simplex convergence tolerance.

    Attributes
    ----------
    initial_level_ : float
        Fitted initial level.
    alpha_ : float
        Fitted smoothing parameter.
    theta_ : float
        Fitted theta parameter.
    fitted_values_ : np.ndarray
        In-sample one-step-ahead fitted values.
    residuals_ : np.ndarray
        In-sample residuals.
    forecast_ : float
        Stored one-step-ahead forecast computed at fit.
    sse_ : float
        Scaled in-sample SSE objective value.
    level_ : float
        Final level state after fitting.
    a_ : float
        Final dynamic line intercept state.
    b_ : float
        Final dynamic line slope state.
    mean_y_ : float
        Final running mean state.

    References
    ----------
    .. [1] Fiorucci, J. A., Pellegrini, T. R., Louzada, F.,
       Petropoulos, F., & Koehler, A. B. (2016). Models for optimising
       the theta method and their relationship to state space models.
       International Journal of Forecasting, 32(4), 1151-1161.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting.stats import DOTM
    >>> y = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1])
    >>> forecaster = DOTM()
    >>> pred = forecaster.iterative_forecast(y, prediction_horizon=2)
    >>> pred.shape
    (2,)
    """

    _tags = {"capability:horizon": False}

    def __init__(
        self,
        initial_level=None,
        alpha=None,
        theta=None,
        initial_level_bounds=(-1e10, 1e10),
        alpha_bounds=(0.1, 0.99),
        theta_bounds=(1.0, 1e10),
        max_iter=500,
        tol=1e-6,
    ):
        self.initial_level = initial_level
        self.alpha = alpha
        self.theta = theta
        self.initial_level_bounds = initial_level_bounds
        self.alpha_bounds = alpha_bounds
        self.theta_bounds = theta_bounds
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit DOTM to a univariate series."""
        y = _prepare_dotm_y(y)
        fixed_mask, fixed_values, lower, upper = self._parameter_arrays()
        x0 = np.array([y[0] / 2.0, 0.5, 2.0], dtype=np.float64)
        for i in range(3):
            if fixed_mask[i]:
                x0[i] = fixed_values[i]
            elif x0[i] < lower[i]:
                x0[i] = lower[i]
            elif x0[i] > upper[i]:
                x0[i] = upper[i]

        params, self.sse_ = _bounded_nelder_mead_dotm(
            x0,
            y,
            fixed_mask,
            fixed_values,
            lower,
            upper,
            float(self.tol),
            int(self.max_iter),
        )

        (
            fitted,
            residuals,
            level,
            a,
            b,
            mean_y,
        ) = _dotm_fitted_values(y, params[0], params[1], params[2])
        forecast = _dotm_forecast(y, 1, params[0], params[1], params[2])[0]

        self.initial_level_ = float(params[0])
        self.alpha_ = float(params[1])
        self.theta_ = float(params[2])
        self.fitted_values_ = fitted
        self.residuals_ = residuals
        self.forecast_ = float(forecast)
        self.level_ = float(level)
        self.a_ = float(a)
        self.b_ = float(b)
        self.mean_y_ = float(mean_y)
        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead from context ``y`` using fitted parameters."""
        y = _prepare_dotm_y(y)
        return float(
            _dotm_forecast(y, 1, self.initial_level_, self.alpha_, self.theta_)[0]
        )

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        """Fit DOTM on ``y`` and recursively forecast ``prediction_horizon`` steps."""
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        if exog is not None:
            raise NotImplementedError("DOTM does not support exog.")
        self.fit(y)
        return _dotm_forecast(
            _prepare_dotm_y(y),
            int(prediction_horizon),
            self.initial_level_,
            self.alpha_,
            self.theta_,
        )

    def _parameter_arrays(self):
        """Return fixed-parameter masks, values, and bounds."""
        fixed_values = np.zeros(3, dtype=np.float64)
        fixed_mask = np.zeros(3, dtype=np.bool_)
        lower = np.array(
            [
                self.initial_level_bounds[0],
                self.alpha_bounds[0],
                self.theta_bounds[0],
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                self.initial_level_bounds[1],
                self.alpha_bounds[1],
                self.theta_bounds[1],
            ],
            dtype=np.float64,
        )
        fixed_params = (self.initial_level, self.alpha, self.theta)
        for i, value in enumerate(fixed_params):
            if value is not None:
                fixed_values[i] = float(value)
                fixed_mask[i] = True
            else:
                fixed_values[i] = np.nan

        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("DOTM parameter bounds must be finite.")
        if np.any(lower > upper):
            raise ValueError("DOTM lower bounds must not exceed upper bounds.")
        for i in range(3):
            if fixed_mask[i] and (
                not np.isfinite(fixed_values[i])
                or fixed_values[i] < lower[i]
                or fixed_values[i] > upper[i]
            ):
                raise ValueError("Fixed DOTM parameters must be finite and in bounds.")
        if lower[1] <= 0.0 or upper[1] >= 1.0:
            raise ValueError("alpha bounds must lie inside (0, 1).")
        if lower[2] < 1.0:
            raise ValueError("theta lower bound must be at least 1.0.")
        return fixed_mask, fixed_values, lower, upper


def _prepare_dotm_y(y):
    """Convert DOTM input to a 1D array and check minimum length."""
    y = np.asarray(y, dtype=np.float64).squeeze()
    if y.shape[0] < 4:
        raise ValueError("DOTM requires at least 4 observations.")
    return y


@njit(cache=True, fastmath=True)
def _dotm_fitted_values(y, initial_level, alpha, theta):
    """Compute DOTM in-sample fitted values and final states."""
    n = y.shape[0]
    fitted = np.empty(n, dtype=np.float64)
    residuals = np.empty(n, dtype=np.float64)
    ell = np.empty(n, dtype=np.float64)
    mean_y = np.empty(n, dtype=np.float64)
    a = np.empty(n, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    omega = 1.0 - 1.0 / theta

    ell[0] = alpha * y[0] + (1.0 - alpha) * initial_level
    mean_y[0] = y[0]
    a[0] = y[0]
    b[0] = 0.0
    fitted[0] = y[0]
    residuals[0] = y[0] - fitted[0]

    for i in range(0, n - 1):
        power = (1.0 - alpha) ** (i + 1)
        if alpha <= 0.0:
            slope_term = 0.0
        else:
            slope_term = b[i] * (1.0 - (1.0 - alpha) ** (i + 2)) / alpha
        fitted[i + 1] = ell[i] + omega * (a[i] * power + slope_term)
        residuals[i + 1] = y[i + 1] - fitted[i + 1]
        ell[i + 1] = alpha * y[i + 1] + (1.0 - alpha) * ell[i]
        mean_y[i + 1] = ((i + 1) * mean_y[i] + y[i + 1]) / (i + 2)
        b[i + 1] = (i * b[i] + 6.0 * (y[i + 1] - mean_y[i]) / (i + 2)) / (i + 3)
        a[i + 1] = mean_y[i + 1] - b[i + 1] * (i + 3) / 2.0

    return fitted, residuals, ell[-1], a[-1], b[-1], mean_y[-1]


@njit(cache=True, fastmath=True)
def _dotm_forecast(y, h, initial_level, alpha, theta):
    """Recursively compute DOTM forecasts beyond the end of ``y``."""
    n = y.shape[0]
    total = n + h
    new_y = np.empty(total, dtype=np.float64)
    for i in range(n):
        new_y[i] = y[i]

    ell = np.empty(total, dtype=np.float64)
    mean_y = np.empty(total, dtype=np.float64)
    a = np.empty(total, dtype=np.float64)
    b = np.empty(total, dtype=np.float64)
    mu = np.empty(total, dtype=np.float64)
    omega = 1.0 - 1.0 / theta

    ell[0] = alpha * new_y[0] + (1.0 - alpha) * initial_level
    mean_y[0] = new_y[0]
    a[0] = new_y[0]
    b[0] = 0.0
    mu[0] = new_y[0]

    for i in range(0, total - 1):
        power = (1.0 - alpha) ** (i + 1)
        slope_term = b[i] * (1.0 - (1.0 - alpha) ** (i + 2)) / alpha
        mu[i + 1] = ell[i] + omega * (a[i] * power + slope_term)
        if i >= n - 1:
            new_y[i + 1] = mu[i + 1]
        ell[i + 1] = alpha * new_y[i + 1] + (1.0 - alpha) * ell[i]
        mean_y[i + 1] = ((i + 1) * mean_y[i] + new_y[i + 1]) / (i + 2)
        b[i + 1] = (i * b[i] + 6.0 * (new_y[i + 1] - mean_y[i]) / (i + 2)) / (i + 3)
        a[i + 1] = mean_y[i + 1] - b[i + 1] * (i + 3) / 2.0

    return mu[n : n + h]


@njit(cache=True, fastmath=True)
def _dotm_sse(params, y, fixed_mask, fixed_values, lower, upper):
    """Scaled DOTM SSE objective with bound penalties."""
    full_params = np.empty(3, dtype=np.float64)
    for i in range(3):
        full_params[i] = fixed_values[i] if fixed_mask[i] else params[i]
        if (
            not np.isfinite(full_params[i])
            or full_params[i] < lower[i]
            or full_params[i] > upper[i]
        ):
            return 1e300
    if full_params[1] <= 0.0 or full_params[1] >= 1.0 or full_params[2] < 1.0:
        return 1e300

    fitted, residuals, _, _, _, _ = _dotm_fitted_values(
        y, full_params[0], full_params[1], full_params[2]
    )
    scale = 0.0
    for i in range(y.shape[0]):
        scale += abs(y[i])
    scale /= y.shape[0]
    if scale <= 1e-12 or not np.isfinite(scale):
        scale = 1.0

    sse = 0.0
    for i in range(2, y.shape[0]):
        if not np.isfinite(fitted[i]) or not np.isfinite(residuals[i]):
            return 1e300
        err = residuals[i] / scale
        sse += err * err
    if not np.isfinite(sse):
        return 1e300
    return sse


@njit(cache=True, fastmath=True)
def _bounded_nelder_mead_dotm(
    x0, y, fixed_mask, fixed_values, lower, upper, tol, max_iter
):
    """Small bounded Nelder-Mead optimiser for DOTM parameters."""
    free_count = 0
    for i in range(3):
        if not fixed_mask[i]:
            free_count += 1

    full_best = np.empty(3, dtype=np.float64)
    for i in range(3):
        full_best[i] = fixed_values[i] if fixed_mask[i] else x0[i]

    if free_count == 0:
        best = _dotm_sse(full_best, y, fixed_mask, fixed_values, lower, upper)
        return full_best, best

    free_idx = np.empty(free_count, dtype=np.int64)
    pos = 0
    for i in range(3):
        if not fixed_mask[i]:
            free_idx[pos] = i
            pos += 1

    simplex = np.empty((free_count + 1, free_count), dtype=np.float64)
    scores = np.empty(free_count + 1, dtype=np.float64)
    for j in range(free_count):
        simplex[0, j] = x0[free_idx[j]]
    for i in range(1, free_count + 1):
        for j in range(free_count):
            simplex[i, j] = simplex[0, j]
        idx = free_idx[i - 1]
        step = 0.05 * abs(simplex[0, i - 1])
        if step <= 1e-6:
            step = 0.05 * (upper[idx] - lower[idx])
        if step <= 1e-6:
            step = 1e-3
        simplex[i, i - 1] += step

    params = np.empty(3, dtype=np.float64)
    for _ in range(max_iter):
        for i in range(free_count + 1):
            for j in range(3):
                params[j] = fixed_values[j] if fixed_mask[j] else 0.0
            for j in range(free_count):
                params[free_idx[j]] = simplex[i, j]
            scores[i] = _dotm_sse(params, y, fixed_mask, fixed_values, lower, upper)

        _sort_simplex(simplex, scores)
        if _simplex_converged(simplex, scores, tol):
            break

        centroid = np.zeros(free_count, dtype=np.float64)
        for i in range(free_count):
            for j in range(free_count):
                centroid[j] += simplex[i, j]
        for j in range(free_count):
            centroid[j] /= free_count

        reflected = centroid + (centroid - simplex[free_count])
        reflected_score = _free_score(
            reflected, y, fixed_mask, fixed_values, lower, upper, free_idx
        )
        if reflected_score < scores[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            expanded_score = _free_score(
                expanded, y, fixed_mask, fixed_values, lower, upper, free_idx
            )
            if expanded_score < reflected_score:
                simplex[free_count] = expanded
                scores[free_count] = expanded_score
            else:
                simplex[free_count] = reflected
                scores[free_count] = reflected_score
        elif reflected_score < scores[free_count - 1]:
            simplex[free_count] = reflected
            scores[free_count] = reflected_score
        else:
            if reflected_score < scores[free_count]:
                contracted = centroid + 0.5 * (reflected - centroid)
            else:
                contracted = centroid + 0.5 * (simplex[free_count] - centroid)
            contracted_score = _free_score(
                contracted, y, fixed_mask, fixed_values, lower, upper, free_idx
            )
            if contracted_score < scores[free_count]:
                simplex[free_count] = contracted
                scores[free_count] = contracted_score
            else:
                for i in range(1, free_count + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])

    for j in range(3):
        full_best[j] = fixed_values[j] if fixed_mask[j] else 0.0
    for j in range(free_count):
        full_best[free_idx[j]] = simplex[0, j]
    best_score = _dotm_sse(full_best, y, fixed_mask, fixed_values, lower, upper)
    return full_best, best_score


@njit(cache=True, fastmath=True)
def _free_score(free_params, y, fixed_mask, fixed_values, lower, upper, free_idx):
    params = np.empty(3, dtype=np.float64)
    for i in range(3):
        params[i] = fixed_values[i] if fixed_mask[i] else 0.0
    for i in range(free_idx.shape[0]):
        params[free_idx[i]] = free_params[i]
    return _dotm_sse(params, y, fixed_mask, fixed_values, lower, upper)


@njit(cache=True, fastmath=True)
def _sort_simplex(simplex, scores):
    n = scores.shape[0]
    for i in range(1, n):
        score = scores[i]
        vertex = simplex[i].copy()
        j = i - 1
        while j >= 0 and scores[j] > score:
            scores[j + 1] = scores[j]
            simplex[j + 1] = simplex[j]
            j -= 1
        scores[j + 1] = score
        simplex[j + 1] = vertex


@njit(cache=True, fastmath=True)
def _simplex_converged(simplex, scores, tol):
    best = scores[0]
    max_score_diff = 0.0
    for i in range(1, scores.shape[0]):
        diff = abs(scores[i] - best)
        if diff > max_score_diff:
            max_score_diff = diff
    max_vertex_diff = 0.0
    for i in range(1, simplex.shape[0]):
        for j in range(simplex.shape[1]):
            diff = abs(simplex[i, j] - simplex[0, j])
            if diff > max_vertex_diff:
                max_vertex_diff = diff
    return max_score_diff <= tol and max_vertex_diff <= tol
