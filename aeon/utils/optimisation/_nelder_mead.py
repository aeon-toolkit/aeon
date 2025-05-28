"""Optimisation algorithms for automatic parameter tuning."""

import numpy as np
from numba import njit


@njit(fastmath=True)
def nelder_mead(
    loss_function,
    num_params,
    data,
    model,
    tol=1e-6,
    max_iter=500,
):
    """
    Perform optimisation using the Nelder–Mead simplex algorithm.

    This function minimises a given loss (objective) function using the Nelder–Mead
    algorithm, a derivative-free method that iteratively refines a simplex of candidate
    solutions. The implementation supports unconstrained minimisation of functions
    with a fixed number of parameters.

    Parameters
    ----------
    loss_function : callable
        The objective function to minimise. Should accept a 1D NumPy array of length
        `num_params` and return a scalar value.
    num_params : int
        The number of parameters (dimensions) in the optimisation problem.
    tol : float, optional (default=1e-6)
        Tolerance for convergence. The algorithm stops when the maximum difference
        between function values at simplex vertices is less than `tol`.
    max_iter : int, optional (default=500)
        Maximum number of iterations to perform.

    Returns
    -------
    best_params : np.ndarray, shape (`num_params`,)
        The parameter vector that minimises the loss function.
    best_value : float
        The value of the loss function at the optimal parameter vector.

    Notes
    -----
    - The initial simplex is constructed by setting each parameter to 0.5,
    with one additional point per dimension at 0.6 for that dimension.
    - This implementation does not support constraints or bounds on the parameters.
    - The algorithm does not guarantee finding a global minimum.

    Examples
    --------
    >>> from numba import njit
    >>> @njit(cache=False, fastmath=True)
    ... def sphere(x, data, model):
    ...     return np.sum(x**2)
    >>> x_opt, val = nelder_mead(sphere, num_params=2, data=None, model=None)
    """
    points = np.full((num_params + 1, num_params), 0.5)
    for i in range(num_params):
        points[i + 1][i] = 0.6
    values = np.array([loss_function(v, data, model) for v in points])
    for _iteration in range(max_iter):
        # Order simplex by function values
        order = np.argsort(values)
        points = points[order]
        values = values[order]

        # Centroid of the best n points
        centre_point = points[:-1].sum(axis=0) / len(points[:-1])

        # Reflection
        # centre + distance between centre and largest value
        reflected_point = centre_point + (centre_point - points[-1])
        reflected_value = loss_function(reflected_point, data, model)
        # if between best and second best, use reflected value
        if len(values) > 1 and values[0] <= reflected_value < values[-2]:
            points[-1] = reflected_point
            values[-1] = reflected_value
            continue
        # Expansion
        # Otherwise if it is better than the best value
        if reflected_value < values[0]:
            expanded_point = centre_point + 2 * (reflected_point - centre_point)
            expanded_value = loss_function(expanded_point, data, model)
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
        contracted_value = loss_function(contracted_point, data, model)
        # If contraction is better use that otherwise move to shrinkage
        if contracted_value < values[-1]:
            points[-1] = contracted_point
            values[-1] = contracted_value
            continue

        # Shrinkage
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = loss_function(points[i], data, model)

        # Convergence check
        if np.max(np.abs(values - values[0])) < tol:
            break
    return points[0], values[0]
