from typing import Optional, Union

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_adtw import _soft_adtw_cost_matrix_with_arrs
from aeon.distances.elastic.soft._soft_distance_utils import _soft_gradient_with_arrs
from aeon.distances.elastic.soft._soft_dtw import _soft_dtw_cost_matrix_with_arrs
from aeon.distances.elastic.soft._soft_erp import _soft_erp_cost_matrix_with_arrs
from aeon.distances.elastic.soft._soft_msm import (
    _soft_msm_cost_matrix_with_arr_independent,
)
from aeon.distances.elastic.soft._soft_shape_dtw import (
    _soft_shape_dtw_cost_matrix_with_arrs,
)
from aeon.distances.elastic.soft._soft_twe import _soft_twe_cost_matrix_with_arrs
from aeon.distances.elastic.soft._soft_wdtw import _soft_wdtw_cost_matrix_with_arrs
from aeon.utils._threading import threaded
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate

MAX_FLOAT = np.finfo(np.float64).max


def gradient_weighted_distance(
    x: np.ndarray,
    y: np.ndarray,
    soft_method: str = "soft_dtw",
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    **kwargs,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
    elif x.ndim == 2 and y.ndim == 2:
        _x = x
        _y = y
    else:
        raise ValueError("x and y must be 1D or 2D")

    bounding_matrix = create_bounding_matrix(
        _x.shape[1], _y.shape[1], window, itakura_max_slope
    )
    return _gradient_weighted_distance(
        _x,
        _y,
        bounding_matrix,
        soft_method,
        gamma,
        **kwargs,
    )


@njit(cache=True, fastmath=True)
def _gradient_weighted_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    soft_method: str,
    gamma: float = 1.0,
    g: float = 0.0,
    nu: float = 0.001,
    lmbda: float = 1.0,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> float:
    x_size = x.shape[1]
    y_size = y.shape[1]

    if soft_method == "soft_dtw":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_dtw_cost_matrix_with_arrs(
            x,
            y,
            bounding_matrix=bounding_matrix,
            gamma=gamma,
        )
    elif soft_method == "soft_wdtw":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_wdtw_cost_matrix_with_arrs(
            x, y, bounding_matrix=bounding_matrix, gamma=gamma, g=g
        )

    elif soft_method == "soft_erp":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_erp_cost_matrix_with_arrs(
            x, y, bounding_matrix=bounding_matrix, gamma=gamma, g=g
        )
    elif soft_method == "soft_twe":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_twe_cost_matrix_with_arrs(
            x,
            y,
            bounding_matrix=bounding_matrix,
            gamma=gamma,
            nu=nu,
            lmbda=lmbda,
        )
    elif soft_method == "soft_msm":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_msm_cost_matrix_with_arr_independent(
            x, y, bounding_matrix=bounding_matrix, gamma=gamma, c=c
        )
    elif soft_method == "soft_shape_dtw":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_shape_dtw_cost_matrix_with_arrs(
            x,
            y,
            bounding_matrix=bounding_matrix,
            gamma=gamma,
            descriptor=descriptor,
            reach=reach,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            transformation_precomputed=transformation_precomputed,
        )
    elif soft_method == "soft_adtw":
        (
            cost_matrix,
            diagonal_arr,
            vertical_arr,
            horizontal_arr,
            diff_dist_matrix,
        ) = _soft_adtw_cost_matrix_with_arrs(
            x,
            y,
            bounding_matrix=bounding_matrix,
            gamma=gamma,
            warp_penalty=warp_penalty,
        )
    else:
        raise ValueError(f"Unknown method: {soft_method}")

    grad = _soft_gradient_with_arrs(
        cost_matrix, diagonal_arr, vertical_arr, horizontal_arr
    )

    weighted_dist = 0.0
    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j] and grad[i, j] > 0:
                sq_dist = diff_dist_matrix[i, j] ** 2
                weighted_dist += sq_dist * grad[i, j]

    return weighted_dist


@threaded
def gradient_weighted_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    soft_method: str = "soft_dtw",
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length_X = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        return _gradient_weighted_pairwise_distance(
            _X,
            soft_method,
            gamma,
            window,
            itakura_max_slope,
            unequal_length_X,
            **kwargs,
        )

    _y, unequal_length_Y = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    unequal_length = unequal_length_X or unequal_length_Y
    return _gradient_weighted_from_multiple_to_multiple_distance(
        _X,
        _y,
        soft_method,
        gamma,
        window,
        itakura_max_slope,
        unequal_length,
        **kwargs,
    )


@njit(cache=True, fastmath=True, parallel=True)
def _gradient_weighted_pairwise_distance(
    X: NumbaList[np.ndarray],
    soft_method: str,
    gamma: float,
    window: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
    g: float = 0.0,
    nu: float = 0.001,
    lmbda: float = 1.0,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    bounding_matrix = None
    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )

    for i in prange(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                _bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            else:
                _bounding_matrix = bounding_matrix

            dist = _gradient_weighted_distance(
                x1,
                x2,
                _bounding_matrix,
                soft_method,
                gamma,
                g,
                nu,
                lmbda,
                c,
                descriptor,
                reach,
                warp_penalty,
                transformation_precomputed,
                transformed_x,
                transformed_y,
            )
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _gradient_weighted_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    soft_method: str,
    gamma: float,
    window: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
    g: float = 0.0,
    nu: float = 0.001,
    lmbda: float = 1.0,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_cases_x = len(x)
    n_cases_y = len(y)
    distances = np.zeros((n_cases_x, n_cases_y))

    bounding_matrix = None
    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )

    for i in prange(n_cases_x):
        for j in range(n_cases_y):
            x1, y1 = x[i], y[j]
            if unequal_length:
                _bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            else:
                _bounding_matrix = bounding_matrix

            distances[i, j] = _gradient_weighted_distance(
                x1,
                y1,
                _bounding_matrix,
                soft_method,
                gamma,
                g,
                nu,
                lmbda,
                c,
                descriptor,
                reach,
                warp_penalty,
                transformation_precomputed,
                transformed_x,
                transformed_y,
            )
    return distances


# if __name__ == "__main__":
#     # Create example time series
#     from aeon.datasets import load_from_ts_file as load_dataset
#     from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
#     from sklearn.metrics import accuracy_score
#
#     DATASET_PATH = "/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts"
#     DATASET_NAME = "GunPoint"
#
#     X, y = load_dataset(full_file_path_and_name=f"{DATASET_PATH}/{DATASET_NAME}/"
#                                                 f"{DATASET_NAME}_TRAIN.ts")
#     X_test, y_test = load_dataset(full_file_path_and_name=f"{DATASET_PATH}/"
#                                                           f"{DATASET_NAME}/"
#                                                           f"{DATASET_NAME}_TEST.ts")
#
#     classifier = KNeighborsTimeSeriesClassifier(
#         n_neighbors=1, distance="gradient_weighted", n_jobs=-1,
#         distance_params={"gamma": 1.0, "soft_method": "soft_dtw"})
#     classifier.fit(X, y)
#
#     print("Fitting the classifier")
#     # Fit the classifier
#     classifier.fit(X, y)
#
#     print("Predicting the labels")
#
#     # Predict the labels
#     y_pred = classifier.predict(X_test)
#     # Calculate the accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#
#     classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1,
#                                                 distance="dtw", n_jobs=-1)
#     classifier.fit(X, y)
#     y_pred = classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"DTW Accuracy: {accuracy}")
#     print(f"DTW Accuracy: {accuracy}")
#
#     classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1,
#                                                 distance="euclidean", n_jobs=-1)
#     classifier.fit(X, y)
#     y_pred = classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Euclidean Accuracy: {accuracy}")
#     print(f"Euclidean Accuracy: {accuracy}")
