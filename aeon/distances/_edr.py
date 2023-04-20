# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

import warnings
from typing import Any, List, Tuple

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._distance_alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances.base import (
    DistanceAlignmentPathCallable,
    DistanceCallable,
    NumbaDistance,
)

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _EdrDistance(NumbaDistance):
    """Edit distance for real sequences (EDR) between two time series.

    ERP was adapted in [1] specifically for distances between trajectories. Like LCSS,
    EDR uses a distance threshold to define when two elements of a series match.
    However, rather than simply count matches and look for the longest sequence,
    ERP applies a (constant) penalty for non-matching elements
    where gaps are inserted to create an optimal alignment.

    References
    ----------
    .. [1] Chen L, Ozsu MT, Oria V: Robust and fast similarity search for moving
    object trajectories. In: Proceedings of the ACM SIGMOD International Conference
    on Management of Data, 2005
    """

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        window: float = None,
        epsilon: float = None,
        **kwargs: Any,
    ) -> DistanceAlignmentPathCallable:
        """Create a no_python compiled edr alignment path distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        window: float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        epsilon : float, defaults = None
            Matching threshold to determine if two subsequences are considered close
            enough to be considered 'common'. If not specified as per the original paper
            epsilon is set to a quarter of the maximum standard deviation.
        kwargs: Any
            Extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]
            No_python compiled edr distance path callable.

        Raises
        ------
        ValueError
            If the input time series are not numpy array.
            If the input time series do not have exactly 2 dimensions.
        """
        _bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)

        if epsilon is not None and not isinstance(epsilon, float):
            raise ValueError("The value of epsilon must be a float.")

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_edr_distance_alignment_path(
                _x: np.ndarray, _y: np.ndarray
            ) -> Tuple[List, float, np.ndarray]:
                if epsilon is None:
                    _epsilon = max(np.std(_x), np.std(_y)) / 4
                else:
                    _epsilon = epsilon
                cost_matrix = _edr_cost_matrix(_x, _y, _bounding_matrix, _epsilon)
                temp_cm = _add_inf_to_out_of_bounds_cost_matrix(
                    cost_matrix, _bounding_matrix
                )
                path = compute_min_return_path(temp_cm)
                distance = float(cost_matrix[-1, -1] / max(_x.shape[1], _y.shape[1]))
                return path, distance, cost_matrix

        else:

            @njit(cache=True)
            def numba_edr_distance_alignment_path(
                _x: np.ndarray, _y: np.ndarray
            ) -> Tuple[List, float]:
                if epsilon is None:
                    _epsilon = max(np.std(_x), np.std(_y)) / 4
                else:
                    _epsilon = epsilon
                cost_matrix = _edr_cost_matrix(_x, _y, _bounding_matrix, _epsilon)
                temp_cm = _add_inf_to_out_of_bounds_cost_matrix(
                    cost_matrix, _bounding_matrix
                )
                path = compute_min_return_path(temp_cm)
                distance = float(cost_matrix[-1, -1] / max(_x.shape[1], _y.shape[1]))
                return path, distance

        return numba_edr_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        epsilon: float = None,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled edr distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        window: float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        epsilon : float, defaults = None
            Matching threshold to determine if two subsequences are considered close
            enough to be considered 'common'. If not specified as per the original paper
            epsilon is set to a quarter of the maximum standard deviation.
        kwargs: Any
            Extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled edr distance callable.

        Raises
        ------
        ValueError
            If the input time series are not numpy array.
            If the input time series do not have exactly 2 dimensions.
          If
        """
        _bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)

        if epsilon is not None and not isinstance(epsilon, float):
            raise ValueError("The value of epsilon must be a float.")

        @njit(cache=True)
        def numba_edr_distance(_x: np.ndarray, _y: np.ndarray) -> float:
            if np.array_equal(_x, _y):
                return 0.0
            if epsilon is None:
                _epsilon = max(np.std(_x), np.std(_y)) / 4
            else:
                _epsilon = epsilon
            cost_matrix = _edr_cost_matrix(_x, _y, _bounding_matrix, _epsilon)
            return float(cost_matrix[-1, -1] / max(_x.shape[1], _y.shape[1]))

        return numba_edr_distance


@njit(cache=True)
def _edr_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: float,
):
    """Compute the edr cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, 2d shape (d (n_dimensions),m (series_length))
        First time series.
    y: np.ndarray, 2d array shape (d, m)
        Second time series.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are infinite values.
    epsilon : float
        Matching threshold to determine if distance between two subsequences are
        considered similar (similar if distance less than the threshold).

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Edr cost matrix between x and y.
    """
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))
    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                curr_dist = 0
                for k in range(dimensions):
                    curr_dist += (x[k][i - 1] - y[k][j - 1]) * (
                        x[k][i - 1] - y[k][j - 1]
                    )
                curr_dist = np.sqrt(curr_dist)
                if curr_dist < epsilon:
                    cost = 0
                else:
                    cost = 1
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + cost,
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                )
    return cost_matrix[1:, 1:]
