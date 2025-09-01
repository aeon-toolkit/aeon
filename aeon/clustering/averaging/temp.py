import numpy as np
from tslearn.barycenters import softdtw_barycenter

from aeon.clustering.averaging._ba_soft import soft_barycenter_average
from aeon.datasets import load_from_ts_file, load_japanese_vowels

DATASET_NAME = "GunPoint"
TOL = 1e-5
GAMMA = 1.0
MAX_ITERS = 50


def tslearn_soft_ba(X):
    _X = X.copy().swapaxes(1, 2)
    return softdtw_barycenter(
        _X,
        gamma=GAMMA,
        tol=TOL,
        max_iter=MAX_ITERS,
    )


if __name__ == "__main__":
    X, y = load_from_ts_file(
        f"/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts/{DATASET_NAME}/{DATASET_NAME}_TRAIN.ts"
    )

    X = X[y == "1"]

    aeon_res = soft_barycenter_average(
        X.copy(),
        gamma=GAMMA,
        tol=TOL,
        max_iters=MAX_ITERS,
        verbose=True,
        distance="soft_msm",
    )
    tslearn_res = tslearn_soft_ba(X)

    temp = aeon_res.copy().swapaxes(0, 1)

    equal = np.allclose(temp, tslearn_res)
    print(f"Equal: {equal}")
