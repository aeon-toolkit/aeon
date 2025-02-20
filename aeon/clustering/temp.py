"""Temp stuff."""

import time

import numpy as np

from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_gunpoint

if __name__ == "__main__":
    # X = make_example_3d_numpy(
    #     n_cases=100, n_channels=1, n_timepoints=1000, random_state=1, return_y=False
    # )
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    averaging_params = {
        "max_iters": 300,
        "tol": 1e-6,
    }

    cluster_arguments = {
        "n_clusters": 2,
        "max_iter": 20,
        "verbose": True,
        "random_state": 1,
        "init": "kmeans++",
        "tol": 1e-6,
        "distance": "msm",
        "averaging_method": "kasba",
        "average_params": averaging_params,
        "n_init": 1,
        "distance_params": {"gamma": 1.0},
    }

    cluster_arguments["n_jobs"] = 1

    kmeans = TimeSeriesKMeans(**cluster_arguments)
    start = time.time()
    original_val = kmeans.fit_predict(X)
    original_inertia = kmeans.inertia_
    end = time.time()
    print("\n++++++++++++++++")  # noqa: T001 T201
    print(f"Original Time taken: {end - start}")  # noqa: T001, T201
    print(f"Original Inertia: {original_inertia}")  # noqa: T001 T201
    print("++++++++++++++++")  # noqa: T001 T201
