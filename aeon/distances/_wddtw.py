# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

import warnings
from typing import Any, List, Tuple

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import DerivativeCallable, average_of_slope
from aeon.distances._distance_alignment_paths import compute_min_return_path
from aeon.distances._numba_utils import is_no_python_compiled_callable
from aeon.distances._wdtw import _weighted_cost_matrix
from aeon.distances.base import (
    DistanceAlignmentPathCallable,
    DistanceCallable,
    NumbaDistance,
)

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _WddtwDistance(NumbaDistance):
    """Weighted derivative dynamic time warping (wddtw) distance between two series.

    Takes the first order derivative, then applies _weighted_cost_matrix to find WDTW
    distance.
    """

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        window: int = None,
        compute_derivative: DerivativeCallable = average_of_slope,
        g: float = 0.0,
        **kwargs: Any,
    ) -> DistanceAlignmentPathCallable:
        """Create a no_python compiled wddtw distance alignment path callable.

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
        window: int, defaults = None
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        compute_derivative: Callable[[np.ndarray], np.ndarray],
                                defaults = average slope difference
            Callable that computes the derivative. If none is provided the average of
            the slope between two points used.
        g: float, defaults = 0.
            Constant that controls the curvature (slope) of the function; that is, g
            controls the level of penalisation for the points with larger phase
            difference.
        kwargs: Any
            Extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]
            No_python compiled wdtw distance path callable.

        Raises
        ------
        ValueError
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the compute derivative callable is not no_python compiled.
            If the value of g is not a float
        """
        _bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)

        if not isinstance(g, float):
            raise ValueError(
                f"The value of g must be a float. The current value is {g}"
            )

        if not is_no_python_compiled_callable(compute_derivative):
            raise ValueError(
                f"The derivative callable must be no_python compiled. The name"
                f"of the callable that must be compiled is "
                f"{compute_derivative.__name__}"
            )

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_wddtw_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float, np.ndarray]:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                cost_matrix = _weighted_cost_matrix(_x, _y, _bounding_matrix, g)
                path = compute_min_return_path(cost_matrix)
                return path, cost_matrix[-1, -1], cost_matrix

        else:

            @njit(cache=True)
            def numba_wddtw_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float]:
                _x = compute_derivative(_x)
                _y = compute_derivative(_y)
                cost_matrix = _weighted_cost_matrix(_x, _y, _bounding_matrix, g)
                path = compute_min_return_path(cost_matrix)
                return path, cost_matrix[-1, -1]

        return numba_wddtw_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int = None,
        compute_derivative: DerivativeCallable = average_of_slope,
        g: float = 0.0,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled wddtw distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        window: int, defaults = None
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        compute_derivative: Callable[[np.ndarray], np.ndarray],
                                defaults = average slope difference
            Callable that computes the derivative. If none is provided the average of
            the slope between two points used.
        g: float, defaults = 0.
            Constant that controls the curvature (slope) of the function; that is, g
            controls the level of penalisation for the points with larger phase
            difference.
        kwargs: Any
            Extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled wddtw distance callable.

        Raises
        ------
        ValueError
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the compute derivative callable is not no_python compiled.
            If the value of g is not a float
        """
        _bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)

        if not isinstance(g, float):
            raise ValueError(
                f"The value of g must be a float. The current value is {g}"
            )

        if not is_no_python_compiled_callable(compute_derivative):
            raise ValueError(
                f"The derivative callable must be no_python compiled. The name"
                f"of the callable that must be compiled is "
                f"{compute_derivative.__name__}"
            )

        @njit(cache=True)
        def numba_wddtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            _x = compute_derivative(_x)
            _y = compute_derivative(_y)
            cost_matrix = _weighted_cost_matrix(_x, _y, _bounding_matrix, g)
            return cost_matrix[-1, -1]

        return numba_wddtw_distance
