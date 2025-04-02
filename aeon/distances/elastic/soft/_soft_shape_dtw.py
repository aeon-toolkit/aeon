r"""Shape Dynamic time warping (ShapeDTW) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._shape_dtw import (
    _get_shape_dtw_distance_from_cost_mat,
    _pad_ts_edges,
    _transform_subsequences,
)
from aeon.distances.elastic.soft._soft_distance_utils import _compute_soft_gradient
from aeon.distances.elastic.soft._soft_dtw import (
    _soft_dtw_cost_matrix,
    _soft_dtw_cost_matrix_with_arrs,
)
from aeon.utils._threading import threaded
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_shape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 15,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))

        x_pad = _pad_ts_edges(x=_x, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )

        return _soft_shape_dtw_distance(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            gamma=gamma,
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = _pad_ts_edges(x=x, reach=reach)
        y_pad = _pad_ts_edges(x=y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )

        return _soft_shape_dtw_distance(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            gamma=gamma,
        )

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_shape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    descriptor: str = "identity",
    reach: int = 15,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
    gamma: float = 1.0,
) -> float:
    # for compilation purposes
    if transformed_x is None:
        transformed_x = x
    if transformed_y is None:
        transformed_y = y

    if not transformation_precomputed:
        new_x = _transform_subsequences(x=x, descriptor=descriptor, reach=reach)
        new_y = _transform_subsequences(x=y, descriptor=descriptor, reach=reach)

        soft_shape_dtw_cost_mat = _soft_dtw_cost_matrix(
            x=new_x, y=new_y, bounding_matrix=bounding_matrix, gamma=gamma
        )
    else:
        soft_shape_dtw_cost_mat = _soft_dtw_cost_matrix(
            x=transformed_x,
            y=transformed_y,
            bounding_matrix=bounding_matrix,
            gamma=gamma,
        )

    return _get_shape_dtw_distance_from_cost_mat(
        x=x, y=y, reach=reach, shape_dtw_cost_mat=soft_shape_dtw_cost_mat
    )


@njit(cache=True, fastmath=True)
def soft_shape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 15,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))

        x_pad = _pad_ts_edges(x=_x, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )

        return _soft_shape_dtw_cost_matrix(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            gamma=gamma,
        )
    if x.ndim == 2 and y.ndim == 2:
        x_pad = _pad_ts_edges(x=x, reach=reach)
        y_pad = _pad_ts_edges(x=y, reach=reach)

        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )

        return _soft_shape_dtw_cost_matrix(
            x=x_pad,
            y=y_pad,
            descriptor=descriptor,
            reach=reach,
            bounding_matrix=bounding_matrix,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            gamma=gamma,
        )

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_shape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    descriptor: str,
    reach: int,
    transformation_precomputed: bool,
    transformed_x: Optional[np.ndarray],
    transformed_y: Optional[np.ndarray],
    gamma: float,
) -> np.ndarray:
    # for compilation purposes
    if transformed_x is None:
        transformed_x = x
    if transformed_y is None:
        transformed_y = y

    if not transformation_precomputed:
        new_x = _transform_subsequences(x=x, descriptor=descriptor, reach=reach)
        new_y = _transform_subsequences(x=y, descriptor=descriptor, reach=reach)

        shapedtw_cost_mat = _soft_dtw_cost_matrix(
            x=new_x, y=new_y, bounding_matrix=bounding_matrix, gamma=gamma
        )
    else:
        shapedtw_cost_mat = _soft_dtw_cost_matrix(
            x=transformed_x,
            y=transformed_y,
            bounding_matrix=bounding_matrix,
            gamma=gamma,
        )

    return shapedtw_cost_mat


@njit(cache=True, fastmath=True)
def soft_shape_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 15,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_shape_dtw_cost_matrix(
        x=x,
        y=y,
        window=window,
        descriptor=descriptor,
        reach=reach,
        itakura_max_slope=itakura_max_slope,
        transformation_precomputed=transformation_precomputed,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
        gamma=gamma,
    )

    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))

        x_pad = _pad_ts_edges(x=_x, reach=reach)
        y_pad = _pad_ts_edges(x=_y, reach=reach)
    elif x.ndim == 2 and y.ndim == 2:
        x_pad = _pad_ts_edges(x=x, reach=reach)
        y_pad = _pad_ts_edges(x=y, reach=reach)
    else:
        raise ValueError("x and y must be 1D or 2D")

    shapedtw_dist = _get_shape_dtw_distance_from_cost_mat(
        x=x_pad, y=y_pad, reach=reach, shape_dtw_cost_mat=cost_matrix
    )

    return compute_min_return_path(cost_matrix), shapedtw_dist


@threaded
def soft_shape_dtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    gamma: float = 1.0,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 15,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_shape_dtw_pairwise_distance(
            X=_X,
            window=window,
            descriptor=descriptor,
            reach=reach,
            itakura_max_slope=itakura_max_slope,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            unequal_length=unequal_length,
            gamma=gamma,
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )

    return _soft_shape_dtw_from_multiple_to_multiple_distance(
        x=_X,
        y=_y,
        window=window,
        descriptor=descriptor,
        reach=reach,
        itakura_max_slope=itakura_max_slope,
        transformation_precomputed=transformation_precomputed,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
        unequal_length=unequal_length,
        gamma=gamma,
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_shape_dtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    descriptor: str,
    reach: int,
    itakura_max_slope: Optional[float],
    transformation_precomputed: bool,
    transformed_x: Optional[np.ndarray],
    unequal_length: bool,
    gamma: float,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in prange(len(X)):
        for j in range(i + 1, n_cases):
            x1_, x2_ = X[i], X[j]
            x1 = _pad_ts_edges(x=x1_, reach=reach)
            x2 = _pad_ts_edges(x=x2_, reach=reach)
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1_.shape[1], x2_.shape[1], window, itakura_max_slope
                )

            if transformation_precomputed and transformed_x is not None:
                transformed_x1 = transformed_x[i]
                transformed_x2 = transformed_x[j]
            else:
                transformed_x1 = x1
                transformed_x2 = x2

            distances[i, j] = _soft_shape_dtw_distance(
                x=x1,
                y=x2,
                descriptor=descriptor,
                reach=reach,
                bounding_matrix=bounding_matrix,
                transformation_precomputed=transformation_precomputed,
                transformed_x=transformed_x1,
                transformed_y=transformed_x2,
                gamma=gamma,
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_shape_dtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    descriptor: str,
    reach: int,
    itakura_max_slope: Optional[float],
    transformation_precomputed: bool,
    transformed_x: Optional[np.ndarray],
    transformed_y: Optional[np.ndarray],
    unequal_length: bool,
    gamma: float,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(m_cases):
            x1_, y1_ = x[i], y[j]
            x1 = _pad_ts_edges(x=x1_, reach=reach)
            y1 = _pad_ts_edges(x=y1_, reach=reach)
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1_.shape[1], y1_.shape[1], window, itakura_max_slope
                )

            if (
                transformation_precomputed
                and transformed_y is not None
                and transformed_x is not None
            ):
                transformed_x1 = transformed_x[i]
                transformed_x2 = transformed_y[j]
            else:
                transformed_x1 = x1
                transformed_x2 = y1

            distances[i, j] = _soft_shape_dtw_distance(
                x=x1,
                y=y1,
                descriptor=descriptor,
                reach=reach,
                bounding_matrix=bounding_matrix,
                transformation_precomputed=transformation_precomputed,
                transformed_x=transformed_x1,
                transformed_y=transformed_x2,
                gamma=gamma,
            )

    return distances


@njit(cache=True, fastmath=True)
def _soft_shape_dtw_cost_matrix_with_arrs(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    descriptor: str = "identity",
    reach: int = 15,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _x = _pad_ts_edges(x=x, reach=reach)
    _y = _pad_ts_edges(x=y, reach=reach)
    if transformed_x is None:
        transformed_x = _x
    if transformed_y is None:
        transformed_y = _y

    if not transformation_precomputed:
        new_x = _transform_subsequences(x=_x, descriptor=descriptor, reach=reach)
        new_y = _transform_subsequences(x=_y, descriptor=descriptor, reach=reach)

        cost_matrix, diagonal_arr, vertical_arr, horizontal_arr, _ = (
            _soft_dtw_cost_matrix_with_arrs(
                x=new_x, y=new_y, bounding_matrix=bounding_matrix, gamma=gamma
            )
        )

    else:
        cost_matrix, diagonal_arr, vertical_arr, horizontal_arr, _ = (
            _soft_dtw_cost_matrix_with_arrs(
                x=transformed_x,
                y=transformed_y,
                bounding_matrix=bounding_matrix,
                gamma=gamma,
            )
        )

    diff_dist_matrix = np.zeros((x.shape[1], y.shape[1]))
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            for k in range(x.shape[0]):
                diff_dist_matrix[i, j] += x[k, i] - y[k, j]

    return cost_matrix, diagonal_arr, vertical_arr, horizontal_arr, diff_dist_matrix


def soft_shape_dtw_gradient(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    descriptor: str = "identity",
    reach: int = 15,
    itakura_max_slope: Optional[float] = None,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    gradient = _compute_soft_gradient(
        x,
        y,
        _soft_shape_dtw_cost_matrix_with_arrs,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        descriptor=descriptor,
        reach=reach,
        transformation_precomputed=transformation_precomputed,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
    )[0]

    dist = soft_shape_dtw_distance(
        x=x,
        y=y,
        gamma=gamma,
        window=window,
        descriptor=descriptor,
        reach=reach,
        itakura_max_slope=itakura_max_slope,
        transformation_precomputed=transformation_precomputed,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
    )

    return gradient, dist
