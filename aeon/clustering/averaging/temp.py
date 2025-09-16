import numpy as np
from tslearn.barycenters import softdtw_barycenter

from aeon.clustering.averaging._ba_soft import soft_barycenter_average
from aeon.datasets import load_from_ts_file
from aeon.transformations.collection import Normalizer

DATASET_NAME = "GunPoint"
TOL = 1e-5
GAMMA = 1.0
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


def auto_soft_msm_params(
    X,
    delta=0.05,  # gate “soft zone” as fraction of typical span
    gamma_mult=0.5,  # gamma ≈ gamma_mult * tau^2
    alpha_bounds=(3.0, 500.0),
    gamma_bounds=(1e-3, 1.0),
):
    X2d = _to_2d_ts(X)
    diffs = np.diff(X2d, axis=1)
    tau = np.median(np.abs(diffs))  # robust scale of first differences
    tau2 = max(tau * tau, 1e-12)

    # gate sharpness (logistic goes 0.01→0.99 over ±s0; s has scale tau^2)
    alpha = 4.6 / (delta * tau2)

    # softmin temperature on squared-diff scale
    gamma = gamma_mult * tau2

    alpha = float(np.clip(alpha, *alpha_bounds))
    gamma = float(np.clip(gamma, *gamma_bounds))
    return alpha, gamma


if __name__ == "__main__":
    X, y = load_from_ts_file(
        f"/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts/{DATASET_NAME}/{DATASET_NAME}_TRAIN.ts"
    )
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    X = X[y == "1"]

    aeon_res_bag = soft_barycenter_average(
        X.copy(),
        gamma=GAMMA,
        tol=TOL,
        max_iters=MAX_ITERS,
        verbose=True,
        distance="soft_bag",
    )

    alpha, gamma = auto_soft_msm_params(X, delta=0.05, gamma_mult=1.5)

    aeon_res_msm = soft_barycenter_average(
        X.copy(),
        gamma=gamma,
        tol=TOL,
        max_iters=MAX_ITERS,
        verbose=True,
        distance="soft_msm",
        alpha=alpha,
    )
    tslearn_res = _tslearn_soft_ba(X)

    temp_bag = aeon_res_bag.copy().swapaxes(0, 1)
    temp_msm = aeon_res_msm.copy().swapaxes(0, 1)

    stop = ""

    # equal = np.allclose(temp, tslearn_res)
    # print(f"Equal: {equal}")
