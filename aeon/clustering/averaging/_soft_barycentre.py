import numpy as np
from numba import set_num_threads
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import _get_init_barycenter
from aeon.distances import create_bounding_matrix
from aeon.distances._distance import ELASTIC_DISTANCE_GRADIENT
from aeon.distances.elastic.soft._soft_adtw import _soft_adtw_cost_matrix_with_arrs
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient_with_diff_dist_matrix,
    _jacobian_product_absolute_distance,
    _jacobian_product_euclidean,
    _jacobian_product_squared_euclidean,
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
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation.collection import _is_numpy_list_multivariate


def _get_soft_distance_cm_with_arrs_function(distance):
    if distance == "soft_dtw":
        return _soft_dtw_cost_matrix_with_arrs, _jacobian_product_squared_euclidean
    if distance == "soft_msm":
        return (
            _soft_msm_cost_matrix_with_arr_independent,
            _jacobian_product_absolute_distance,
        )
    if distance == "soft_twe":
        return _soft_twe_cost_matrix_with_arrs, _jacobian_product_euclidean
    if distance == "soft_adtw":
        return _soft_adtw_cost_matrix_with_arrs, _jacobian_product_squared_euclidean
    if distance == "soft_shape_dtw":
        return (
            _soft_shape_dtw_cost_matrix_with_arrs,
            _jacobian_product_squared_euclidean,
        )
    if distance == "soft_wdtw":
        return _soft_wdtw_cost_matrix_with_arrs, _jacobian_product_squared_euclidean
    if distance == "soft_erp":
        return _soft_erp_cost_matrix_with_arrs, _jacobian_product_euclidean

    raise ValueError(
        f"Invalid distance: {distance}. Must be one " f"{ELASTIC_DISTANCE_GRADIENT}"
    )


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
    tol=1e-3,
    max_iters=50,
    init_barycenter="mean",
    distance="soft_dtw",
    random_state=None,
    verbose=False,
    n_jobs=1,
    **kwargs,
):
    n_jobs = check_n_jobs(n_jobs)
    set_num_threads(n_jobs)
    if len(X) <= 1:
        return X

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    barycenter = _get_init_barycenter(
        _X,
        init_barycenter,
        distance,
        random_state=random_state,
        **kwargs,
    )
    _X, unequal_length, bounding_matrix, barycenter, kwargs = (
        _preprocess_arrays_and_kwargs(_X, distance, barycenter, **kwargs)
    )

    soft_cm_with_arrs_func, jacobian_function = (
        _get_soft_distance_cm_with_arrs_function(distance)
    )

    n_timepoints = barycenter.shape[1]

    def _func(Z):
        G = np.zeros_like(barycenter)
        total_distance = 0
        _Z = Z.reshape(1, n_timepoints)

        for i in range(len(_X)):
            _bounding_matrix = bounding_matrix
            if unequal_length:
                _bounding_matrix = create_bounding_matrix(
                    _X[i].shape[1], n_timepoints, **kwargs
                )
            # grad, value, diff_dist_matrix = soft_gradient_func(
            #       _Z, _X[i], gamma=gamma, **kwargs
            #   )
            grad, value, diff_dist_matrix = (
                _compute_soft_gradient_with_diff_dist_matrix(
                    _Z,
                    _X[i],
                    cost_matrix_with_arrs_func=soft_cm_with_arrs_func,
                    gamma=gamma,
                    bounding_matrix=_bounding_matrix,
                    **kwargs,
                )
            )

            jacobian_product = jacobian_function(_Z, _X[i], grad, diff_dist_matrix)
            G += jacobian_product
            total_distance += value

        return total_distance, G

    res = minimize(
        _func,
        barycenter.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iters, disp=verbose),
    )

    # TWE is padded with a 0 at the start so remove first element
    if distance == "soft_twe":
        return res.x.reshape(*barycenter.shape)[:, 1:]

    return res.x.reshape(*barycenter.shape)


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
