"""Generic distance factories and decorators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate


def make_pairwise_self(core_distance: Callable):
    """Specialised X-vs-X pairwise kernel for a distance core.

    core_distance signature:
        (x2d: np.ndarray, y2d: np.ndarray, *params) -> float
    """

    @njit(cache=True, fastmath=True, parallel=True)
    def _pairwise_self(
        X: NumbaList[np.ndarray],
        *params,
    ) -> np.ndarray:
        n_cases = len(X)
        distances = np.zeros((n_cases, n_cases))

        for i in prange(n_cases):
            for j in range(i + 1, n_cases):
                d = core_distance(X[i], X[j], *params)
                distances[i, j] = d
                distances[j, i] = d

        return distances

    return _pairwise_self


def make_pairwise_x_to_y(core_distance: Callable):

    @njit(cache=True, fastmath=True, parallel=True)
    def _pairwise_x_to_y(
        X: NumbaList[np.ndarray],
        Y: NumbaList[np.ndarray],
        *params,
    ) -> np.ndarray:
        n_cases = len(X)
        m_cases = len(Y)
        distances = np.zeros((n_cases, m_cases))

        for i in prange(n_cases):
            for j in range(m_cases):
                distances[i, j] = core_distance(X[i], Y[j], *params)

        return distances

    return _pairwise_x_to_y


def build_distance(*, core_distance: Callable, name: str):
    """Build a numba distance function with automatic 1D/2D handling.

    Parameters
    ----------
    core_distance : Callable
        Core distance function with signature:
        (x2d: np.ndarray, y2d: np.ndarray, *params) -> float
        Both inputs are guaranteed to be 2D (n_channels, n_timepoints).
    name : str
        Name for the distance function (used in __name__).

    Returns
    -------
    Callable
        Numba JIT-compiled distance function with signature:
        (x, y, *params) -> float
    """

    @njit(cache=True, fastmath=True)
    def distance(x, y, *params):
        # Handle 1D inputs by reshaping to 2D (1, n_timepoints)
        # Note: separate variables needed for Numba type unification
        if x.ndim == 1:
            x_2d = x.reshape((1, x.shape[0]))
        elif x.ndim == 2:
            x_2d = x
        else:
            raise ValueError("x must be 1D or 2D")

        if y.ndim == 1:
            y_2d = y.reshape((1, y.shape[0]))
        elif y.ndim == 2:
            y_2d = y
        else:
            raise ValueError("y must be 1D or 2D")

        return core_distance(x_2d, y_2d, *params)

    distance.__name__ = f"{name}_distance"
    return distance


def build_pairwise_distance(
    *,
    core_distance: Callable,
    name: str,
):
    """Build the public pairwise_distance wrapper (threaded + conversion).

    Parameters
    ----------
    core_distance : Callable
        Distance function to use for pairwise computation.
    name : str
        Name for the pairwise function (used in __name__).

    Returns
    -------
    Callable
        Threaded pairwise distance function with signature:
        pairwise_distance(X, y=None, n_jobs=1, *params) -> np.ndarray
    """
    pairwise_self = make_pairwise_self(core_distance)
    pairwise_x_to_y = make_pairwise_x_to_y(core_distance)

    @threaded
    def pairwise_distance(
        X: np.ndarray | list[np.ndarray],
        y: np.ndarray | list[np.ndarray] | None = None,
        n_jobs: int = 1,
        *params,
    ) -> np.ndarray:
        multivariate_conversion = _is_numpy_list_multivariate(X, y)
        _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)

        if y is None:
            return pairwise_self(_X, *params)

        _Y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
        return pairwise_x_to_y(_X, _Y, *params)

    pairwise_distance.__name__ = f"{name}_pairwise_distance"
    return pairwise_distance


def distance(name: str):
    """Create a public distance function from a core implementation.

    The decorated function should:
    - Accept 2D arrays (n_channels, n_timepoints)
    - Return float distance
    - Have a complete docstring

    Creates a public function that handles both 1D and 2D inputs.

    Parameters
    ----------
    name : str
        Base name for the distance (e.g., "euclidean" creates
        "euclidean_distance")

    Returns
    -------
    Callable
        Decorator function that wraps the core implementation
    """

    def decorator(core_func: Callable) -> Callable:
        # Get docstring from core function
        core_doc = core_func.__doc__

        @njit(cache=True, fastmath=True)
        def _wrapper(x, y, *params):
            # Convert 1D to 2D
            if x.ndim == 1:
                x_2d = x.reshape((1, x.shape[0]))
            elif x.ndim == 2:
                x_2d = x
            else:
                raise ValueError("x must be 1D or 2D")

            if y.ndim == 1:
                y_2d = y.reshape((1, y.shape[0]))
            elif y.ndim == 2:
                y_2d = y
            else:
                raise ValueError("y must be 1D or 2D")

            return core_func(x_2d, y_2d, *params)

        # Set public name and docstring
        public_name = f"{name}_distance"
        _wrapper.__name__ = public_name
        _wrapper.__doc__ = core_doc

        # Store for later retrieval
        _wrapper._is_distance_wrapper = True
        _wrapper._core_func = core_func

        return _wrapper

    return decorator


def pairwise(name: str):
    """Create a public pairwise distance function.

    Creates a threaded pairwise distance function that handles collections.

    Parameters
    ----------
    name : str
        Base name for the distance

    Returns
    -------
    Callable
        Decorator function
    """

    def decorator(distance_func: Callable) -> Callable:
        # Get the docstring from the distance function
        distance_doc = distance_func.__doc__

        # Build the pairwise kernels
        pairwise_self = make_pairwise_self(distance_func)
        pairwise_x_to_y = make_pairwise_x_to_y(distance_func)

        @threaded
        def pairwise_distance(
            X: np.ndarray | list[np.ndarray],
            y: np.ndarray | list[np.ndarray] | None = None,
            n_jobs: int = 1,
            *params,
        ) -> np.ndarray:
            multivariate_conversion = _is_numpy_list_multivariate(X, y)
            _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)

            if y is None:
                return pairwise_self(_X, *params)

            _Y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
            return pairwise_x_to_y(_X, _Y, *params)

        # Set name but don't override doc (we'll add a note)
        public_name = f"{name}_pairwise_distance"
        pairwise_distance.__name__ = public_name

        # Simple docstring reference
        if distance_doc:
            pairwise_distance.__doc__ = f"""Pairwise version of {name}_distance.

Computes pairwise distances between collections of time series.

See {name}_distance for distance computation details.
"""

        # Store reference to distance function
        pairwise_distance._distance_func = distance_func

        return pairwise_distance

    return decorator
