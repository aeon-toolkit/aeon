"""Temp stuff."""

import os

import numpy as np

from aeon.clustering.averaging import elastic_barycenter_average
from aeon.datasets import load_from_ts_file
from aeon.transformations.collection import Normalizer


def _load_data(dataset_name, path_to_data, normalize=True):
    path_to_train_data = os.path.join(
        path_to_data, f"{dataset_name}/{dataset_name}_TRAIN.ts"
    )
    path_to_test_data = os.path.join(
        path_to_data, f"{dataset_name}/{dataset_name}_TEST.ts"
    )
    X_train, y_train = load_from_ts_file(path_to_train_data)
    X_test, y_test = load_from_ts_file(path_to_test_data)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    if normalize:
        scaler = Normalizer()
        X = scaler.fit_transform(X)
    return X, y


if __name__ == "__main__":
    DATASET_PATH = "/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts"
    dataset = "GunPoint"
    # X = make_example_3d_numpy(
    #     n_cases=10, n_channels=1, n_timepoints=10, random_state=1, return_y=False
    # )
    X, y = _load_data(dataset, DATASET_PATH, normalize=False)
    verbose = False
    # for distance in VALID_SOFT_BA_DISTANCE_METHODS:
    distance = "soft_erp"
    print(f"Starting experiment now with distance: {distance}")  # noqa: T001, T201
    from aeon.clustering.averaging import mean_average

    barycenter, cost = elastic_barycenter_average(
        X,
        method="soft",
        verbose=verbose,
        distance=distance,
        random_state=1,
        return_cost=True,
        tol=1e-6,
        max_iters=20,
        n_jobs=-1,
        ba_subset_size=0.5,
        decay_rate=0.01,
        initial_step_size=0.05,
        final_step_size=0.005,
    )
    swap = barycenter.swapaxes(0, 1)
    mean_avg = mean_average(X)
    means_swap = mean_avg.swapaxes(0, 1)
    print(f"Equal: {np.allclose(barycenter, mean_avg)}")  # noqa: T001, T201
    # X = make_example_3d_numpy(
    #     n_cases=100, n_channels=1, n_timepoints=1000, random_state=1, return_y=False
    # )
    # # X = make_example_3d_numpy(
    # #   n_cases=10, n_channels=1, n_timepoints=10, random_state=1, return_y=False
    # # )
    # # X, y = load_gunpoint(split='train')
    # # X = X[y == '1']
    #
    # n_jobs = -1
    # kwargs = {
    #     "max_iters": 30,
    #     "tol": 1e-5,
    # }
    #
    # start = time.time()
    # kasba_avg = elastic_barycenter_average(
    #     X,
    #     method="kasba",
    #     verbose=verbose,
    #     distance=distance,
    #     n_jobs=n_jobs,
    #     random_state=2,
    #     **kwargs,
    # )
    # end = time.time()
    # print(f"KASBA Time taken: {end - start}")  # noqa: T001, T201
    #
    # kasba_1_job = elastic_barycenter_average(
    #     X,
    #     method="kasba",
    #     verbose=verbose,
    #     distance=distance,
    #     n_jobs=1,
    #     random_state=2,
    #     **kwargs,
    # )
    # print(f"KASBA Equal: {np.allclose(kasba_avg, kasba_1_job)}")  # noqa: T001 T201
    #
    # start = time.time()
    # subgradient_avg = elastic_barycenter_average(
    #     X,
    #     method="subgradient",
    #     verbose=verbose,
    #     distance=distance,
    #     n_jobs=n_jobs,
    #     random_state=1,
    #     **kwargs,
    # )
    # end = time.time()
    # print(f"Subgradient Time taken: {end - start}")  # noqa: T001, T201
    #
    # subgradient_1_job = elastic_barycenter_average(
    #     X,
    #     method="subgradient",
    #     verbose=verbose,
    #     distance=distance,
    #     n_jobs=1,
    #     random_state=1,
    #     **kwargs,
    # )
    # print(  # noqa: T001 T201
    #     f"Subgradient Equal: "  # noqa: T001 T201
    #     f"{np.allclose(subgradient_avg, subgradient_1_job)}"  # noqa: T001 T201
    # )  # noqa: T001 T201
    #
    # start = time.time()
    # petitjean_avg = elastic_barycenter_average(
    #     X,
    #     method="petitjean",
    #     verbose=verbose,
    #     distance=distance,
    #     n_jobs=n_jobs,
    #     random_state=1,
    #     **kwargs,
    # )
    # end = time.time()
    # print(f"Petitjean Time taken: {end - start}")  # noqa: T001, T201
    #
    # petitjean_1_job = elastic_barycenter_average(
    #     X,
    #     method="petitjean",
    #     verbose=verbose,
    #     distance=distance,
    #     n_jobs=1,
    #     random_state=1,
    #     **kwargs,
    # )
    # print(  # noqa: T001 T201
    #     f"Petitjean Equal: "  # noqa: T001 T201
    #     f"{np.allclose(petitjean_avg, petitjean_1_job)}"  # noqa: T001 T201
    # )  # noqa: T001 T201
    #
    # start = time.time()
    # soft_avg = elastic_barycenter_average(
    #     X,
    #     method="soft",
    #     gamma=1.0,
    #     verbose=verbose,
    #     distance=f"soft_{distance}",
    #     n_jobs=n_jobs,
    #     random_state=1,
    #     **kwargs,
    # )
    # end = time.time()
    # print(f"Soft Time taken: {end - start}")  # noqa: T001, T201
    # soft_1_job = elastic_barycenter_average(
    #     X,
    #     method="soft",
    #     gamma=1.0,
    #     verbose=verbose,
    #     distance=f"soft_{distance}",
    #     n_jobs=1,
    #     random_state=1,
    #     **kwargs,
    # )
    # print(f"Soft Equal: {np.allclose(soft_avg, soft_1_job)}")  # noqa: T001 T201

    # _X = X.swapaxes(1, 2)
    # start = time.time()
    # tslearn_avg = softdtw_barycenter(_X, gamma=1.0, max_iter=30, tol=1e-3)
    # end = time.time()
    # print(f"Tslearn Time taken: {end - start}")
    # print(f"Equal: {np.allclose(soft_avg, tslearn_avg)}")
