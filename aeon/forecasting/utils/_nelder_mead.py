"""Optimisation algorithms for automatic parameter tuning."""

import numpy as np
from numba import njit

from aeon.forecasting.utils._extract_paras import _ets_is_damped, _ets_num_params
from aeon.forecasting.utils._loss_functions import _arima_fit, _ets_fit


@njit(cache=True)
def dispatch_loss(fn_id, params, data, model):
    if fn_id == 0:
        value = _arima_fit(params, data, model)
    elif fn_id == 1:
        value = _ets_fit(params, data, model)[0]
    else:
        raise ValueError("Unknown loss function ID")
    if np.isnan(value) or np.isinf(value):
        return np.inf
    return value


@njit(cache=True, fastmath={"contract"})
def _init_arima_simplex(points, data, model, simplex_init):
    num_params = points.shape[1]
    points[:] = simplex_init
    for i in range(num_params):
        points[i + 1, i] = simplex_init * 1.2
    if model[0] == 1:
        # The constant is on the scale of the data while the AR/MA coefficients
        # are O(1). Started near zero with a tiny step, the simplex needed
        # hundreds of expansions to reach it and fits routinely stopped at
        # max_iter far from the optimum. Start it at the series mean, with a
        # step on the scale of the data.
        mean = np.mean(data)
        points[:, 0] = mean
        points[1, 0] = mean + 0.1 * np.std(data) + 1e-3


@njit(cache=True, fastmath=True)
def _init_ets_simplex(points, model):
    alpha_idx = 0
    beta_idx = 1
    phi_idx = 2
    error_type = model[0]
    trend_type = model[1]
    seasonality_type = model[2]
    damped = trend_type != 0 and _ets_is_damped(model)
    points[:] = 0.0
    base = points[0]
    # Multiplicative-error/additive-seasonal models are especially sensitive to
    # invalid fitted states from a large initial level/seasonal smoothing value.
    base_alpha = 0.1 if error_type == 2 and seasonality_type == 1 else 0.5
    base_level = 0.01 if error_type == 2 and seasonality_type == 1 else 0.1
    base[alpha_idx] = base_alpha
    param_index = 1
    if trend_type != 0:
        base[param_index] = base_level
        param_index += 1
        if damped:
            base[param_index] = 0.98
            param_index += 1
    if seasonality_type != 0:
        base[param_index] = base_level

    for i in range(1, len(points)):
        # Numba row assignment copies the base vertex before perturbing one slot.
        points[i] = base
        param = i - 1
        if param == alpha_idx:
            points[i, param] = base_alpha * 1.2
        elif trend_type != 0 and param == beta_idx:
            points[i, param] = base_level * 1.2
        elif damped and param == phi_idx:
            points[i, param] = 0.9
        else:
            points[i, param] = base_level * 1.2


@njit(cache=True, fastmath={"contract"})
def nelder_mead(
    loss_id,
    num_params,
    data,
    model,
    tol=1e-6,
    max_iter=500,
    simplex_init=0.5,
):
    """
    Perform optimisation using the Nelder–Mead simplex algorithm.

    This function minimises a given loss (objective) function using the Nelder–Mead
    algorithm, a derivative-free method that iteratively refines a simplex of candidate
    solutions. The implementation supports unconstrained minimisation of functions
    with a fixed number of parameters.

    Parameters
    ----------
    loss_id : int
        ID for loss function to optimise, used by ``dispatch_loss``.
    num_params : int
        The number of parameters (dimensions) in the optimisation problem.
    data : np.ndarray
        The input data used by the loss function. The shape and content depend on the
        specific loss function being minimised.
    model : np.ndarray
        The model or context in which the loss function operates. This could be any
        other object that the ``loss_function`` requires to compute its value.
        The exact type and structure of ``model`` should be compatible with the
        `loss_function`.
    tol : float, default=1e-6
        Tolerance for convergence. The algorithm stops when the maximum difference
        between function values at simplex vertices is less than ``tol`` and all
        vertices are within a relative 1e-6 of the best vertex. This is checked
        every iteration.
    max_iter : int, default=500
        Maximum number of iterations to perform.
    simplex_init : float, default=0.5
        Initial value for the simplex search matrix.

    Returns
    -------
    best_params : np.ndarray, shape (`num_params`,)
        The parameter vector that minimises the loss function.
    best_value : float
        The value of the loss function at the optimal parameter vector.

    Notes
    -----
    - The initial simplex is constructed by setting each parameter to simplex_init (
    default to 0.5), with one additional point per dimension at simplex_init*1.2
    (0.6 by default) for that dimension. For ARIMA models with a constant, the
    constant instead starts at the mean of ``data``, with a step of 10% of its
    standard deviation. ETS models use their own starting simplex.
    - This implementation does not support constraints or bounds on the parameters.
    - The algorithm does not guarantee finding a global minimum and may occasionally
    converge on a poor solution.

    References
    ----------
    .. [1] Nelder, J. A. and Mead, R. (1965).
       A Simplex Method for Function Minimization.
       The Computer Journal, 7(4), 308–313.
       https://doi.org/10.1093/comjnl/7.4.308
    """
    points = np.full((num_params + 1, num_params), simplex_init)
    if loss_id == 1:
        _init_ets_simplex(points, model)
    else:
        _init_arima_simplex(points, data, model, simplex_init)
    return _nelder_mead_from_simplex(loss_id, points, data, model, tol, max_iter)


@njit(cache=True, fastmath={"contract"})
def _simplex_converged(points, values, tol):
    """Check whether a sorted simplex has converged.

    The values at all vertices must be within ``tol`` of the best value, and
    every vertex within a relative 1e-6 of the best vertex in each coordinate.
    Equal values alone are not enough: Nelder-Mead can pass through a simplex
    with near equal values that is still spread out (on a ridge or against a
    constraint) and then continue to a much better point.
    """
    if not np.isfinite(values[0]):
        return False
    if np.max(np.abs(values - values[0])) >= tol:
        return False
    for i in range(1, len(points)):
        for j in range(points.shape[1]):
            if abs(points[i, j] - points[0, j]) > 1e-6 * (1.0 + abs(points[0, j])):
                return False
    return True


@njit(cache=True, fastmath={"contract"})
def _nelder_mead_from_simplex(loss_id, points, data, model, tol=1e-6, max_iter=500):
    """Run Nelder-Mead from the starting simplex ``points``.

    ``points`` has shape ``(num_params + 1, num_params)``, one vertex per row,
    and is not modified. Callers can use this to restart the search from their
    own starting points. See :func:`nelder_mead` for the other parameters and
    the return values.
    """
    points = points.copy()
    values = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        values[i] = dispatch_loss(loss_id, points[i].copy(), data, model)
    for _ in range(max_iter):
        # Order simplex by function values
        order = np.argsort(values)
        points = points[order]
        values = values[order]

        # Convergence check
        if _simplex_converged(points, values, tol):
            break

        # Centroid of the best n points
        centre_point = points[:-1].sum(axis=0) / len(points[:-1])

        # Reflection
        # centre + distance between centre and largest value
        reflected_point = centre_point + (centre_point - points[-1])
        reflected_value = dispatch_loss(loss_id, reflected_point, data, model)
        # if between best and second best, use reflected value
        if (
            len(values) > 1
            and not np.isnan(reflected_value)
            and values[0] <= reflected_value < values[-2]
        ):
            points[-1] = reflected_point
            values[-1] = reflected_value
            continue
        # Expansion
        # Otherwise if it is better than the best value
        if not np.isnan(reflected_value) and reflected_value < values[0]:
            expanded_point = centre_point + 2 * (reflected_point - centre_point)
            expanded_value = dispatch_loss(loss_id, expanded_point, data, model)
            # if less than reflected value use expanded, otherwise go back to reflected
            if not np.isnan(expanded_value) and expanded_value < reflected_value:
                points[-1] = expanded_point
                values[-1] = expanded_value
            else:
                points[-1] = reflected_point
                values[-1] = reflected_value
            continue
        # Contraction
        # Otherwise if reflection is worse than all current values
        contracted_point = centre_point - 0.5 * (centre_point - points[-1])
        contracted_value = dispatch_loss(loss_id, contracted_point, data, model)
        # If contraction is better use that otherwise move to shrinkage
        if not np.isnan(contracted_value) and contracted_value < values[-1]:
            points[-1] = contracted_point
            values[-1] = contracted_value
            continue

        # Shrinkage
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = dispatch_loss(loss_id, points[i], data, model)

    # ``values`` is kept in sync with ``points`` at every exit of the main
    # loop (sort, reflect/expand/contract write to both, shrink updates both
    # per vertex). The previous implementation re-evaluated ``dispatch_loss``
    # for all n+1 simplex vertices to pick the best, which was a wasted
    # n+1 loss evaluations per optimiser call.
    best = np.argmin(values)
    return points[best], values[best]


@njit(cache=True, fastmath=True)
def _ets_param_bounds(x, model):
    """Return the valid interval of each ETS parameter, others held at ``x``.

    Matches ``_ets_params_are_valid``: 0 <= alpha < 1, 0 <= beta < alpha,
    0 < phi <= 1 and 0 <= gamma < 1 - alpha. The bounds of alpha therefore
    depend on beta and gamma, and those of beta and gamma on alpha.
    """
    k = len(x)
    lower = np.zeros(k)
    upper = np.ones(k)
    i = 1
    beta_idx = -1
    gamma_idx = -1
    if model[1] != 0:
        beta_idx = i
        i += 1
        if _ets_is_damped(model):
            i += 1
    if model[2] != 0:
        gamma_idx = i
    if beta_idx >= 0:
        lower[0] = x[beta_idx]
        upper[beta_idx] = x[0]
    if gamma_idx >= 0:
        upper[0] = 1.0 - x[gamma_idx]
        upper[gamma_idx] = 1.0 - x[0]
    return lower, upper


@njit(cache=True, fastmath=True)
def _ets_simplex_around(x0, model):
    """Build a starting simplex with ``x0`` as one vertex.

    Each other vertex moves one parameter: phi down by 0.08, the others by half
    their value plus 0.01, upwards if that stays valid, else downwards, else
    half way towards the further bound of the parameter's valid interval. If
    that interval is empty (``x0`` is itself on or outside a bound) the
    parameter moves up by 0.01, so the simplex never collapses.
    """
    k = len(x0)
    lower, upper = _ets_param_bounds(x0, model)
    phi_idx = 2 if model[1] != 0 and _ets_is_damped(model) else -1
    points = np.empty((k + 1, k))
    for i in range(k + 1):
        points[i] = x0
    for i in range(k):
        x = x0[i]
        if i == phi_idx:
            new = x - 0.08 if x - 0.08 > 0.0 else 0.5 * (x + 1.0)
        else:
            step = 0.5 * x + 0.01
            if x + step < upper[i]:
                new = x + step
            elif x - step > lower[i]:
                new = x - step
            else:
                far = upper[i] if upper[i] - x >= x - lower[i] else lower[i]
                new = 0.5 * (x + far)
        if new == x:
            new = x + 0.01
        points[i + 1, i] = new
    return points


@njit(cache=True, fastmath=True)
def _ets_r_style_start(model):
    """Return a starting point in the style of R's forecast::ets.

    alpha = 0.2, beta = 0.1 * alpha, phi = 0.978 and gamma = 0.05 * (1 - alpha).
    """
    x0 = np.empty(_ets_num_params(model))
    x0[0] = 0.2
    i = 1
    if model[1] != 0:
        x0[i] = 0.02
        i += 1
        if _ets_is_damped(model):
            x0[i] = 0.978
            i += 1
    if model[2] != 0:
        x0[i] = 0.04
    return x0


@njit(cache=True, fastmath={"contract"})
def _fit_ets_parameters(data, model, max_iter=200, tol=1e-6):
    """Fit ETS smoothing parameters by minimising the AIC with Nelder-Mead.

    The ETS likelihood surface often has several local minima, and a single
    Nelder-Mead run from a fixed starting simplex regularly stops in a poor
    one. Run from two different starting points (aeon's default simplex and an
    R-style start with small smoothing parameters), then restart from the best
    result, and return the best of the three runs.

    Parameters
    ----------
    data : np.ndarray
        The series being fitted.
    model : np.ndarray
        ETS model array, see ``_ets_fit``.
    max_iter : int, default=200
        Maximum Nelder-Mead iterations per run.
    tol : float, default=1e-6
        Convergence tolerance per run.

    Returns
    -------
    best_params : np.ndarray
        Smoothing parameters, laid out as in ``_extract_ets_params``.
    best_value : float
        AIC at ``best_params``.
    """
    k = _ets_num_params(model)
    points = np.empty((k + 1, k))
    _init_ets_simplex(points, model)
    best_x, best_v = _nelder_mead_from_simplex(1, points, data, model, tol, max_iter)
    x, v = _nelder_mead_from_simplex(
        1,
        _ets_simplex_around(_ets_r_style_start(model), model),
        data,
        model,
        tol,
        max_iter,
    )
    if v < best_v:
        best_x, best_v = x, v
    x, v = _nelder_mead_from_simplex(
        1, _ets_simplex_around(best_x, model), data, model, tol, max_iter
    )
    if v < best_v:
        best_x, best_v = x, v
    return best_x, best_v


@njit(cache=True, fastmath={"contract"})
def _lstsq(X, y):
    """Least squares coefficients, NaN if the problem is underdetermined."""
    if X.shape[0] <= X.shape[1]:
        return np.full(X.shape[1], np.nan)
    return np.linalg.lstsq(X, y)[0]


@njit(cache=True, fastmath={"contract"})
def _hannan_rissanen(data, model):
    """Hannan-Rissanen estimates of an ARMA model, as ARIMA starting values.

    Fit a long autoregression by least squares to estimate the innovations,
    then regress the series on its own lags and the lagged innovations [1]_.
    For a pure AR model this is the least squares fit, which is the conditional
    sum of squares optimum. Returns ``[c?, phi..., theta...]`` in the layout of
    ``_extract_arma_params``, or NaNs if the series is too short.

    References
    ----------
    .. [1] E. J. Hannan and J. Rissanen, Recursive estimation of mixed
       autoregressive-moving average order. Biometrika, 69(1), 81-94, 1982.
    """
    c, p, q = model[0], model[1], model[2]
    n = len(data)
    out = np.full(c + p + q, np.nan)
    if q == 0:
        resid = np.zeros(0)
        start = p
    else:
        # long autoregression, of order 10 log10(n) as in R's ar()
        m = int(min(max(p + q, np.ceil(10.0 * np.log10(n))), n // 4))
        if m < 1:
            return out
        X = np.ones((n - m, m + 1))
        for j in range(m):
            X[:, j + 1] = data[m - j - 1 : n - j - 1]
        coef = _lstsq(X, data[m:])
        if np.isnan(coef[0]):
            return out
        resid = np.zeros(n)
        resid[m:] = data[m:] - X @ coef
        start = m + max(p, q)
    if n - start <= c + p + q:
        return out
    X = np.ones((n - start, c + p + q))
    for j in range(p):
        X[:, c + j] = data[start - j - 1 : n - j - 1]
    for j in range(q):
        X[:, c + p + j] = resid[start - j - 1 : n - j - 1]
    return _lstsq(X, data[start:])


@njit(cache=True, fastmath={"contract"})
def _arima_simplex_around(x0, data, model):
    """Build a starting simplex with ``x0`` as one vertex.

    Each other vertex moves one parameter: the constant by 10% of the standard
    deviation of ``data``, AR and MA coefficients by 0.1.
    """
    k = len(x0)
    points = np.empty((k + 1, k))
    for i in range(k + 1):
        points[i] = x0
    for i in range(k):
        if model[0] == 1 and i == 0:
            points[i + 1, i] += 0.1 * np.std(data) + 1e-3
        else:
            points[i + 1, i] += 0.1
    return points


@njit(cache=True, fastmath={"contract"})
def _fit_arima_parameters(data, model, max_iter=200, simplex_init=0.1, tol=1e-6):
    """Fit ARMA parameters by minimising the AIC with Nelder-Mead.

    The conditional sum of squares surface of higher order ARMA models has
    several local minima. Run Nelder-Mead from the default starting simplex
    (constant at the series mean) and from Hannan-Rissanen estimates, then
    restart from the best result, and return the best of the three runs.

    Parameters
    ----------
    data : np.ndarray
        The (differenced) series being fitted.
    model : np.ndarray
        ``[constant, p, q]``.
    max_iter : int, default=200
        Maximum Nelder-Mead iterations per run.
    simplex_init : float, default=0.1
        Starting value of the AR and MA coefficients in the default simplex.
    tol : float, default=1e-6
        Convergence tolerance per run.

    Returns
    -------
    best_params : np.ndarray
        Parameters, laid out as in ``_extract_arma_params``.
    best_value : float
        AIC at ``best_params``.
    """
    k = model[0] + model[1] + model[2]
    points = np.empty((k + 1, k))
    _init_arima_simplex(points, data, model, simplex_init)
    best_x, best_v = _nelder_mead_from_simplex(0, points, data, model, tol, max_iter)
    if k == 0:
        return best_x, best_v
    x0 = _hannan_rissanen(data, model)
    if np.all(np.isfinite(x0)):
        x, v = _nelder_mead_from_simplex(
            0, _arima_simplex_around(x0, data, model), data, model, tol, max_iter
        )
        if v < best_v:
            best_x, best_v = x, v
    x, v = _nelder_mead_from_simplex(
        0, _arima_simplex_around(best_x, data, model), data, model, tol, max_iter
    )
    if v < best_v:
        best_x, best_v = x, v
    return best_x, best_v
