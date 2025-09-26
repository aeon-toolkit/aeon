import numpy as np
from tslearn.barycenters import softdtw_barycenter

from aeon.clustering.averaging._ba_soft import soft_barycenter_average
from aeon.datasets import load_from_ts_file
from aeon.transformations.collection import Normalizer

DATASET_NAME = "GunPoint"
TOL = 1e-5
GAMMA = 0.1
MAX_ITERS = 50


def _tslearn_soft_ba(X):
    _X = X.copy().swapaxes(1, 2)
    return softdtw_barycenter(
        _X,
        gamma=GAMMA,
        tol=TOL,
        max_iter=MAX_ITERS,
    )


def _to_2d_ts(X):
    X = np.asarray(X)
    if X.ndim == 3:  # (n_cases, n_channels, n_timepoints)
        return X.mean(axis=1)
    if X.ndim == 2:  # (n_cases, n_timepoints)
        return X
    raise ValueError(
        "X must be (n_cases, n_timepoints) or (n_cases, n_channels, n_timepoints)"
    )


if __name__ == "__main__":
    import time

    X, y = load_from_ts_file(
        f"/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts/{DATASET_NAME}/{DATASET_NAME}_TRAIN.ts"
    )
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    X = X[y == "1"]

    # aeon_res_bag = soft_barycenter_average(
    #     X.copy(),
    #     gamma=GAMMA,
    #     tol=TOL,
    #     max_iters=MAX_ITERS,
    #     verbose=True,
    #     distance="soft_bag",
    # )

    tslearn_res = _tslearn_soft_ba(X)
    aeon_res_msm = soft_barycenter_average(
        X.copy(),
        gamma=GAMMA,
        tol=TOL,
        max_iters=MAX_ITERS,
        verbose=True,
        distance="soft_divergence_msm",
    )
    # temp_bag = aeon_res_bag.copy().swapaxes(0, 1)
    temp_res = aeon_res_msm.copy().swapaxes(0, 1)

    equal = np.allclose(temp_res, tslearn_res)
    print(f"Equal: {equal}")
    stop = ""
