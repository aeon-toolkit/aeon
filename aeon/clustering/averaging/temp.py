"""Temp stuff."""

import time

from aeon.clustering.averaging import elastic_barycenter_average
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":
    verbose = False
    distance = "dtw"

    X = make_example_3d_numpy(
        n_cases=100, n_channels=1, n_timepoints=1000, random_state=1, return_y=False
    )
    # X = make_example_3d_numpy(
    #   n_cases=10, n_channels=1, n_timepoints=10, random_state=1, return_y=False
    # )
    # X, y = load_gunpoint(split='train')
    # X = X[y == '1']

    n_jobs = -1
    kwargs = {
        "max_iters": 30,
        "tol": 1e-5,
    }

    start = time.time()
    kasba_avg = elastic_barycenter_average(
        X,
        method="kasba",
        verbose=verbose,
        distance=distance,
        n_jobs=n_jobs,
        random_state=2,
        **kwargs,
    )
    kasba_avg = kasba_avg.swapaxes(0, 1)
    end = time.time()
    print(f"KASBA Time taken: {end - start}")  # noqa: T001, T201

    start = time.time()
    subgradient_avg = elastic_barycenter_average(
        X,
        method="subgradient",
        verbose=verbose,
        distance=distance,
        n_jobs=n_jobs,
        random_state=1,
        **kwargs,
    )
    subgradient_avg = subgradient_avg.swapaxes(0, 1)
    end = time.time()
    print(f"Subgradient Time taken: {end - start}")  # noqa: T001, T201

    start = time.time()
    petitjean_avg = elastic_barycenter_average(
        X,
        method="petitjean",
        verbose=verbose,
        distance=distance,
        n_jobs=n_jobs,
        random_state=1,
        **kwargs,
    )
    petitjean_avg = petitjean_avg.swapaxes(0, 1)
    end = time.time()
    print(f"Petitjean Time taken: {end - start}")  # noqa: T001, T201

    start = time.time()
    soft_avg = elastic_barycenter_average(
        X,
        method="soft",
        gamma=1.0,
        verbose=verbose,
        distance=f"soft_{distance}",
        n_jobs=n_jobs,
        random_state=1,
        **kwargs,
    )
    end = time.time()
    print(f"Soft Time taken: {end - start}")  # noqa: T001, T201
    soft_avg = soft_avg.swapaxes(0, 1)

    # _X = X.swapaxes(1, 2)
    # start = time.time()
    # tslearn_avg = softdtw_barycenter(_X, gamma=1.0, max_iter=30, tol=1e-3)
    # end = time.time()
    # print(f"Tslearn Time taken: {end - start}")
    # print(f"Equal: {np.allclose(soft_avg, tslearn_avg)}")
