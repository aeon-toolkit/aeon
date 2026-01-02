"""Optimisation algorithms for automatic parameter tuning."""

import numpy as np
from numba import njit

from aeon.forecasting.utils._loss_functions import _arima_fit, _ets_fit


@njit(cache=True, fastmath=True)
def dispatch_loss(fn_id, params, data, model):
    if fn_id == 0:
        return _arima_fit(params, data, model)
    if fn_id == 1:
        return _ets_fit(params, data, model)[0]
    else:
        raise ValueError("Unknown loss function ID")


@njit(cache=True, fastmath=True)
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
        between function values at simplex vertices is less than ``tol``.
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
    (0.6 by default) for that dimension.
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
    for i in range(num_params):
        points[i + 1][i] = simplex_init * 1.2
    values = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        values[i] = dispatch_loss(loss_id, points[i].copy(), data, model)
    for _ in range(max_iter):
        # Order simplex by function values
        order = np.argsort(values)
        points = points[order]
        values = values[order]

        # Centroid of the best n points
        centre_point = points[:-1].sum(axis=0) / len(points[:-1])

        # Reflection
        # centre + distance between centre and largest value
        reflected_point = centre_point + (centre_point - points[-1])
        reflected_value = dispatch_loss(loss_id, reflected_point, data, model)
        # if between best and second best, use reflected value
        if len(values) > 1 and values[0] <= reflected_value < values[-2]:
            points[-1] = reflected_point
            values[-1] = reflected_value
            continue
        # Expansion
        # Otherwise if it is better than the best value
        if reflected_value < values[0]:
            expanded_point = centre_point + 2 * (reflected_point - centre_point)
            expanded_value = dispatch_loss(loss_id, expanded_point, data, model)
            # if less than reflected value use expanded, otherwise go back to reflected
            if expanded_value < reflected_value:
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
        if contracted_value < values[-1]:
            points[-1] = contracted_point
            values[-1] = contracted_value
            continue

        # Shrinkage
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = dispatch_loss(loss_id, points[i], data, model)

        # Convergence check
        if np.max(np.abs(values - values[0])) < tol:
            break
    return points[0], values[0]
