import warnings
from typing import Optional, Union

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import (
    VALID_SOFT_BA_DISTANCE_METHODS,
    _ba_setup,
)
from aeon.distances import create_bounding_matrix
from aeon.distances.elastic.soft._soft_adtw import _soft_adtw_cost_matrix_with_arrs
from aeon.distances.elastic.soft._soft_distance_utils import (
    _jacobian_product_absolute_distance,
    _jacobian_product_euclidean,
    _jacobian_product_squared_euclidean,
    _soft_gradient_with_arrs,
)
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
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


def _preprocess_arrays_and_kwargs(X, distance, barycenter, **kwargs):
    """Preprocess the arrays and kwargs for the soft barycenter averaging."""
    if distance == "soft_twe":
        X = np.pad(
            X, pad_width=((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0
        )
        barycenter = np.pad(
            barycenter, pad_width=((0, 0), (1, 0)), mode="constant", constant_values=0
        )

    multivariate_conversion = _is_numpy_list_multivariate(X)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if multivariate_conversion:
        raise ValueError(
            "Multivariate time series are not supported for soft barycenter averaging"
        )

    if "bounding_matrix" not in kwargs:
        bounding_matrix = create_bounding_matrix(
            _X[0].shape[1],
            _X[1].shape[1],
            window=kwargs.get("window"),
            itakura_max_slope=kwargs.get("itakura_max_slope"),
        )
    else:
        bounding_matrix = kwargs.pop("bounding_matrix")

    return _X, unequal_length, bounding_matrix, barycenter, kwargs


def soft_barycenter_average(
    X,
    gamma=1.0,
    weights=None,
    method="L-BFGS-B",
    tol=1e-5,
    max_iters=50,
    init_barycenter="mean",
    previous_cost: Optional[float] = None,
    previous_distance_to_center: Optional[np.ndarray] = None,
    distance="soft_dtw",
    random_state=None,
    verbose=False,
    n_jobs=1,
    return_distances_to_center: bool = False,
    return_cost: bool = False,
    **kwargs,
):
    if len(X) <= 1:
        center = X[0] if X.ndim == 3 else X
        if return_distances_to_center and return_cost:
            return center, np.zeros(X.shape[0]), 0.0
        elif return_distances_to_center:
            return center, np.zeros(X.shape[0])
        elif return_cost:
            return center, 0.0
        return center

    if distance not in VALID_SOFT_BA_DISTANCE_METHODS:
        raise ValueError(
            f"Invalid distance method: {distance}. Valid method are: "
            f"{VALID_SOFT_BA_DISTANCE_METHODS}"
        )

    (
        _X,
        barycenter,
        prev_barycenter,
        cost,
        previous_cost,
        distances_to_center,
        previous_distance_to_center,
        random_state,
        n_jobs,
    ) = _ba_setup(
        X,
        distance=distance,
        init_barycenter=init_barycenter,
        previous_cost=previous_cost,
        previous_distance_to_center=previous_distance_to_center,
        precomputed_medoids_pairwise_distance=None,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    _X, unequal_length, bounding_matrix, barycenter, kwargs = (
        _preprocess_arrays_and_kwargs(_X, distance, barycenter, **kwargs)
    )
    if weights is None:
        weights = np.ones(len(_X))

    n_timepoints = barycenter.shape[1]
    bounding_matrix = create_bounding_matrix(
        n_timepoints, n_timepoints, window=kwargs.get("window")
    )
    distances_to_center = None

    def _func(Z):
        total_distance, jacobian_product, distances_to_center = (
            _soft_barycenter_one_iter(
                barycenter=Z.reshape(*barycenter.shape),
                X=_X,
                bounding_matrix=bounding_matrix,
                unequal_length=unequal_length,
                n_timepoints=n_timepoints,
                distance=distance,
                gamma=gamma,
                **kwargs,
            )
        )
        return total_distance, jacobian_product

    res = minimize(
        _func,
        barycenter.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iters, disp=verbose, maxls=40),
    )

    if res.success is False:
        warnings.warn(
            f"Optimisation failed to converge."
            f"Reason given by method: {res.message}. For more detail set "
            f"verbose=True.",
            RuntimeWarning,
            stacklevel=2,
        )
        if verbose:
            print(  # noqa: T201
                f"Failed to converge for {distance} barycenter "  # noqa: T201
                f"averaging. With gamma = {gamma}. The reason given is: "  # noqa: T201
                f"{res.message}"  # noqa: T201
            )  # noqa: T201

    # TWE is padded with a 0 at the start so remove first element
    final_barycenter = res.x.reshape(*barycenter.shape)
    if distance == "soft_twe":
        final_barycenter = final_barycenter[:, 1:]

    if return_distances_to_center and return_cost:
        return final_barycenter, distances_to_center, res.fun
    elif return_distances_to_center:
        return final_barycenter, distances_to_center
    elif return_cost:
        return final_barycenter, res.fun
    return final_barycenter


@njit(cache=True, fastmath=True, parallel=True)
def _soft_barycenter_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    bounding_matrix: np.ndarray,
    unequal_length: bool,
    n_timepoints: int,
    distance: str = "msm",
    window: Union[float, None] = None,
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
    gamma: float = 1.0,
):
    X_size = len(X)
    local_jacobian_products = np.zeros(
        (X_size, barycenter.shape[0], barycenter.shape[1])
    )
    local_distances = np.zeros(X_size)
    distances_to_center = np.zeros(X_size)

    for i in prange(X_size):
        _bounding_matrix = bounding_matrix
        curr_ts = X[i]
        if unequal_length:
            _bounding_matrix = create_bounding_matrix(
                curr_ts.shape[1], n_timepoints, window=window
            )
        if distance == "soft_dtw":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_dtw_cost_matrix_with_arrs(
                barycenter,
                curr_ts,
                bounding_matrix=_bounding_matrix,
                gamma=gamma,
            )
        elif distance == "soft_wdtw":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_wdtw_cost_matrix_with_arrs(
                barycenter, curr_ts, bounding_matrix=bounding_matrix, gamma=gamma, g=g
            )
        elif distance == "soft_erp":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_erp_cost_matrix_with_arrs(
                barycenter, curr_ts, bounding_matrix=bounding_matrix, gamma=gamma, g=g
            )
        elif distance == "soft_twe":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_twe_cost_matrix_with_arrs(
                barycenter,
                curr_ts,
                bounding_matrix=bounding_matrix,
                gamma=gamma,
                nu=nu,
                lmbda=lmbda,
            )
        elif distance == "soft_msm":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_msm_cost_matrix_with_arr_independent(
                barycenter, curr_ts, bounding_matrix=bounding_matrix, gamma=gamma, c=c
            )
        elif distance == "soft_shape_dtw":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_shape_dtw_cost_matrix_with_arrs(
                barycenter,
                curr_ts,
                bounding_matrix=bounding_matrix,
                gamma=gamma,
                descriptor=descriptor,
                reach=reach,
                transformed_x=transformed_x,
                transformed_y=transformed_y,
                transformation_precomputed=transformation_precomputed,
            )
        elif distance == "soft_adtw":
            (
                cost_matrix,
                diagonal_arr,
                vertical_arr,
                horizontal_arr,
                diff_dist_matrix,
            ) = _soft_adtw_cost_matrix_with_arrs(
                barycenter,
                curr_ts,
                bounding_matrix=bounding_matrix,
                gamma=gamma,
                warp_penalty=warp_penalty,
            )

        grad = _soft_gradient_with_arrs(
            cost_matrix, diagonal_arr, vertical_arr, horizontal_arr
        )
        curr_dist = cost_matrix[-1, -1]
        local_distances[i] = curr_dist
        distances_to_center[i] = curr_dist

        if distance == "soft_msm":
            local_jacobian_products[i] = _jacobian_product_absolute_distance(
                barycenter, curr_ts, grad, diff_dist_matrix
            )
        elif distance == "soft_twe" or distance == "soft_erp":
            local_jacobian_products[i] = _jacobian_product_euclidean(
                barycenter, curr_ts, grad, diff_dist_matrix
            )
        else:
            local_jacobian_products[i] = _jacobian_product_squared_euclidean(
                barycenter, curr_ts, grad, diff_dist_matrix
            )

    # Combine results after parallel section
    jacobian_product = np.zeros_like(barycenter)
    total_distance = 0.0
    for i in range(X_size):
        jacobian_product += local_jacobian_products[i]
        total_distance += local_distances[i]

    return total_distance, jacobian_product, distances_to_center


if __name__ == "__main__":
    import time

    from tslearn.barycenters import softdtw_barycenter

    from aeon.datasets import load_gunpoint as load_dataset

    class_labels_to_load = "1"

    gamma = 0.1
    verbose = False
    X, y = load_dataset()
    X = X[y == class_labels_to_load]

    aeon_start = time.time()
    dtw_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, verbose=verbose, distance="soft_dtw", tol=1e-3
    )
    aeon_end = time.time()
    print(f"Aeon soft-DBA time: {aeon_end - aeon_start}")  # noqa: T201
    dtw_ba = dtw_ba.swapaxes(0, 1)

    tslearn_X = X.swapaxes(1, 2)
    tslearn_start = time.time()
    tslearn_ba = softdtw_barycenter(tslearn_X, gamma=gamma, max_iter=50, tol=1e-3)
    tslearn_end = time.time()
    print(f"Tslearn soft-DBA time: {tslearn_end - tslearn_start}")  # noqa: T201

    print(f"Soft DTW barycenter equal: {np.allclose(dtw_ba, tslearn_ba)}")  # noqa: T201

    twe_aeon_start = time.time()
    twe_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_twe", verbose=verbose
    )
    twe_aeon_end = time.time()
    twe_ba = twe_ba.swapaxes(0, 1)
    print(f"Aeon twe time: {twe_aeon_end - twe_aeon_start}")  # noqa: T201

    msm_aeon_start = time.time()
    msm_ba = soft_barycenter_average(
        X,
        max_iters=50,
        gamma=gamma,
        distance="soft_msm",
        verbose=verbose,
        tol=1e-3,
        c=1.0,
    )
    msm_aeon_end = time.time()
    print(f"Aeon soft-MBA time: {msm_aeon_end - msm_aeon_start}")  # noqa: T201
    msm_ba = msm_ba.swapaxes(0, 1)

    adtw_aeon_start = time.time()
    adtw_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_adtw", verbose=verbose
    )
    adtw_aeon_end = time.time()
    adtw_ba = adtw_ba.swapaxes(0, 1)
    print(f"Aeon adtw time: {adtw_aeon_end - adtw_aeon_start}")  # noqa: T201

    shape_dtw_aeon_start = time.time()
    shape_dtw_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_shape_dtw", verbose=verbose
    )
    shape_dtw_aeon_end = time.time()
    shape_dtw_ba = shape_dtw_ba.swapaxes(0, 1)
    print(  # noqa: T201
        f"Aeon shape dtw time: "  # noqa: T201
        f"{shape_dtw_aeon_end - shape_dtw_aeon_start}"  # noqa: T201
    )  # noqa: T201

    wdtw_aeon_start = time.time()
    wdtw_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_wdtw", verbose=verbose
    )
    wdtw_aeon_end = time.time()
    wdtw_ba = wdtw_ba.swapaxes(0, 1)
    print(f"Aeon wdtw time: {wdtw_aeon_end - wdtw_aeon_start}")  # noqa: T201

    erp_aeon_start = time.time()
    erp_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_erp", verbose=verbose
    )
    erp_aeon_end = time.time()
    erp_ba = erp_ba.swapaxes(0, 1)
    print(f"Aeon erp time: {erp_aeon_end - erp_aeon_start}")  # noqa: T201

    # twe_ba_01 = soft_barycenter_average(
    #   X, max_iters=50, gamma=0.1, distance="soft_twe"
    #   )
    # twe_ba_01 = twe_ba_01.swapaxes(0, 1)
    # twe_ba_1 = soft_barycenter_average(
    #   X, max_iters=50, gamma=1.0, distance="soft_twe"
    #   )
    # twe_ba_1 = twe_ba_1.swapaxes(0, 1)
    # # print(f"Soft DTW barycenter equal: {np.allclose(dtw_ba, tslearn_ba)}")
    # # print(f"Soft TWE barycenter equal: {np.allclose(twe_ba, tslearn_ba)}")
    # average = X.mean(axis=0)
    stop = ""
