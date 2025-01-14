import numpy as np
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import _get_init_barycenter
from aeon.distances._distance import DISTANCES, ELASTIC_DISTANCE_GRADIENT
from aeon.distances.elastic.soft._soft_distance_utils import (
    _jacobian_product_absolute_distance,
    _jacobian_product_squared_euclidean,
)


def _get_soft_gradient_function(distance):
    if distance in ELASTIC_DISTANCE_GRADIENT:
        for dist in DISTANCES:
            if dist["name"] == distance:
                return dist["gradient"]

    raise ValueError(
        f"Invalid distance: {distance}. Must be one " f"{ELASTIC_DISTANCE_GRADIENT}"
    )


def soft_barycenter_average(
    X,
    gamma=1.0,
    weights=None,
    method="L-BFGS-B",
    tol=1e-6,
    max_iters=50,
    init_barycenter="mean",
    distance="soft_dtw",
    random_state=None,
    verbose=False,
    **kwargs,
):
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

    soft_gradient_func = _get_soft_gradient_function(distance)

    n_timepoints = barycenter.shape[1]

    jacobian_function = _jacobian_product_squared_euclidean
    if distance == "soft_msm":
        jacobian_function = _jacobian_product_absolute_distance

    def _func(Z):
        G = np.zeros_like(barycenter)
        total_distance = 0
        _Z = Z.reshape(1, n_timepoints)

        for i in range(len(_X)):
            grad, value = soft_gradient_func(_Z, _X[i], gamma=gamma, **kwargs)
            jacobian_product = jacobian_function(_Z, X[i], grad)
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

    return res.x.reshape(*barycenter.shape)


if __name__ == "__main__":
    import time

    from aeon.datasets import load_gunpoint

    gamma = 1.0
    X, y = load_gunpoint()
    X = X[y == "1"]
    # aeon_start = time.time()
    # dtw_ba = soft_barycenter(X, max_iters=50, gamma=gamma, verbose=True)
    # aeon_end = time.time()
    # print(f"Aeon dtw time: {aeon_end - aeon_start}")  # noqa: T201
    # dtw_ba = dtw_ba.swapaxes(0, 1)
    #
    twe_aeon_start = time.time()
    twe_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_twe", verbose=False
    )
    twe_aeon_end = time.time()
    twe_ba = twe_ba.swapaxes(0, 1)
    print(f"Aeon twe time: {twe_aeon_end - twe_aeon_start}")  # noqa: T201

    msm_aeon_start = time.time()
    msm_ba = soft_barycenter_average(
        X, max_iters=50, gamma=gamma, distance="soft_msm", verbose=False
    )
    msm_aeon_end = time.time()
    print(f"Aeon msm time: {msm_aeon_end - msm_aeon_start}")  # noqa: T201
    msm_ba = msm_ba.swapaxes(0, 1)
    twe_ba_01 = soft_barycenter_average(X, max_iters=50, gamma=0.1, distance="soft_twe")
    twe_ba_01 = twe_ba_01.swapaxes(0, 1)
    twe_ba_1 = soft_barycenter_average(X, max_iters=50, gamma=1.0, distance="soft_twe")
    twe_ba_1 = twe_ba_1.swapaxes(0, 1)
    # print(f"Soft DTW barycenter equal: {np.allclose(dtw_ba, tslearn_ba)}")
    # print(f"Soft TWE barycenter equal: {np.allclose(twe_ba, tslearn_ba)}")
    average = X.mean(axis=0)
    stop = ""
