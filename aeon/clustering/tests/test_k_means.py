"""Tests for time series k-means."""
import numpy as np
import pytest
from sklearn import metrics
from sklearn.utils import check_random_state

from aeon.clustering._k_means import TimeSeriesKMeans
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.distances import euclidean_distance
from aeon.utils.validation._dependencies import _check_estimator_deps

expected_results = {
    "mean": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "dba": [1, 0, 1, 0, 0],
}

expected_train_result = {"mean": 0.47307692307692306, "dba": 0.6}

expected_score = {"mean": 0.3192307692307692, "dba": 0.4}

expected_iters = {"mean": 6, "dba": 3}

expected_labels = {
    "mean": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
    ],
    "dba": [0, 1, 0, 0, 0],
}


@pytest.mark.skipif(
    not _check_estimator_deps(TimeSeriesKMeans, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kmeans():
    """Test implementation of Kmeans."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmeans = TimeSeriesKMeans(
        averaging_method="mean",
        random_state=1,
        n_init=2,
        n_clusters=2,
        init_algorithm="kmeans++",
        distance="euclidean",
    )
    train_predict = kmeans.fit_predict(X_train)
    train_mean_score = metrics.rand_score(y_train, train_predict)

    test_mean_result = kmeans.predict(X_test)
    mean_score = metrics.rand_score(y_test, test_mean_result)
    proba = kmeans.predict_proba(X_test)

    assert np.array_equal(test_mean_result, expected_results["mean"])
    assert mean_score == expected_score["mean"]
    assert train_mean_score == expected_train_result["mean"]
    assert kmeans.n_iter_ == expected_iters["mean"]
    assert np.array_equal(kmeans.labels_, expected_labels["mean"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (40, 2)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


@pytest.mark.skipif(
    not _check_estimator_deps(TimeSeriesKMeans, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kmeans_dba():
    """Test implementation of Kmeans using dba."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    num_test_values = 5

    kmeans = TimeSeriesKMeans(
        averaging_method="ba",
        random_state=1,
        n_init=2,
        n_clusters=2,
        init_algorithm="kmeans++",
        distance="dtw",
    )
    train_predict = kmeans.fit_predict(X_train[0:num_test_values])
    train_mean_score = metrics.rand_score(y_train[0:num_test_values], train_predict)

    test_mean_result = kmeans.predict(X_test[0:num_test_values])
    mean_score = metrics.rand_score(y_test[0:num_test_values], test_mean_result)
    proba = kmeans.predict_proba(X_test[0:num_test_values])

    assert np.array_equal(test_mean_result, expected_results["dba"])
    assert mean_score == expected_score["dba"]
    assert train_mean_score == expected_train_result["dba"]
    assert kmeans.n_iter_ == expected_iters["dba"]
    assert np.array_equal(kmeans.labels_, expected_labels["dba"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (5, 2)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _get_model_centres(data, distance, average_params=None, distance_params=None):
    """Get the centres of a model."""
    model = TimeSeriesKMeans(
        averaging_method="ba",
        random_state=1,
        n_init=2,
        n_clusters=2,
        init_algorithm="random",
        distance=distance,
        average_params=average_params,
        distance_params=distance_params,
    )
    model.fit(data)
    return model.cluster_centers_


def test_different_ba():
    """Test different ba methods."""
    X_train, y_train = load_basic_motions(split="train")

    num_test_values = 5
    data = X_train[0:num_test_values]

    # Test passing distance param
    default_dba = _get_model_centres(data, distance="dtw")
    dba_specified = _get_model_centres(
        data, distance="dtw", average_params={"distance": "dtw"}
    )
    assert np.array_equal(dba_specified, default_dba)

    # Test another distance and check passing custom distance param
    default_mba = _get_model_centres(data, distance="msm")
    mba_specified = _get_model_centres(
        data, distance="msm", average_params={"distance": "msm"}
    )
    assert np.array_equal(mba_specified, default_mba)
    assert not np.array_equal(mba_specified, dba_specified)

    # Test passing custom parameter
    mba_custom_params = _get_model_centres(
        data, distance="msm", average_params={"independent": False}
    )
    assert not np.array_equal(mba_custom_params, default_mba)

    mba_custom_params_window = _get_model_centres(
        data, distance="msm", average_params={"window": 0.2}
    )

    assert not np.array_equal(mba_custom_params, mba_custom_params_window)

    # Test passing multiple params
    mba_custom_params_window_and_indep = _get_model_centres(
        data, distance="msm", average_params={"window": 0.2, "independent": False}
    )

    # Check not equal to just when indep set
    assert not np.array_equal(mba_custom_params, mba_custom_params_window_and_indep)
    # Check not equal to just when window set
    assert not np.array_equal(
        mba_custom_params_window_and_indep, mba_custom_params_window
    )
    # Check not equal to default
    assert not np.array_equal(default_mba, mba_custom_params_window_and_indep)


def check_value_in_every_cluster(num_clusters, initial_centres):
    """Check that every cluster has at least one value."""
    original_length = len(initial_centres)
    assert original_length == num_clusters

    # Check no duplicates
    for i in range(len(initial_centres)):
        curr = initial_centres[i]
        for j in range(len(initial_centres)):
            if i == j:
                continue
            other = initial_centres[j]
            assert not np.array_equal(curr, other)


def test_means_init():
    """Test implementation of Kmedoids."""
    X_train, _ = load_gunpoint(split="train")
    custom_init_centres = X_train[[12, 13]]
    X_train = X_train[:10]

    num_clusters = 4
    kmeans = TimeSeriesKMeans(
        random_state=1,
        n_init=1,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
        n_clusters=num_clusters,
    )
    kmeans._random_state = check_random_state(kmeans.random_state)
    kmeans._distance_params = {}
    kmeans._distance_callable = euclidean_distance
    first_mean_result = kmeans._first_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, first_mean_result)
    random_mean_result = kmeans._random_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, random_mean_result)
    kmean_plus_plus_result = kmeans._kmeans_plus_plus_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, kmean_plus_plus_result)

    # Test setting manual init centres
    num_clusters = 2
    kmeans = TimeSeriesKMeans(
        random_state=1,
        n_init=1,
        max_iter=5,
        init_algorithm=custom_init_centres,
        distance="euclidean",
        n_clusters=num_clusters,
    )
    kmeans.fit(X_train)

    assert np.array_equal(kmeans.cluster_centers_, custom_init_centres)


def test_custom_distance_params():
    X_train, y_train = load_basic_motions(split="train")

    num_test_values = 10
    data = X_train[0:num_test_values]

    # Test passing distance param
    default_dist = _get_model_centres(data, distance="msm")
    custom_params_dist = _get_model_centres(
        data, distance="msm", distance_params={"window": 0.2}
    )
    assert not np.array_equal(default_dist, custom_params_dist)
