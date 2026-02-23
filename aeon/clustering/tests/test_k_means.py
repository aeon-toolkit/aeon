"""Test time series k-means clustering."""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn import metrics

from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_basic_motions
from aeon.distances._distance import ELASTIC_DISTANCES
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.expected_results.expected_clustering_results import (
    k_means_expected_results,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING
from aeon.testing.utils._distance_parameters import (
    TEST_DISTANCE_WITH_CUSTOM_DISTANCE,
    TEST_DISTANCE_WITH_PARAMS,
    TEST_DISTANCES_WITH_FULL_ALIGNMENT_PATH,
)


def _run_kmeans_test(kmeans_params, n_cases, n_channels, n_timepoints):
    X_train = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=1,
        return_y=False,
    )
    X_test = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=2,
        return_y=False,
    )

    kmeans = TimeSeriesKMeans(**kmeans_params)
    n_clusters = kmeans_params["n_clusters"]
    train_predict = kmeans.fit_predict(X_train)

    assert isinstance(train_predict, np.ndarray)
    assert train_predict.shape == (n_cases,)
    assert np.unique(train_predict).shape[0] <= n_clusters
    assert kmeans.cluster_centers_.shape == (n_clusters, n_channels, n_timepoints)

    test_result = kmeans.predict(X_test)

    assert isinstance(test_result, np.ndarray)
    assert test_result.shape == (n_cases,)
    assert np.unique(test_result).shape[0] <= n_clusters

    proba = kmeans.predict_proba(X_test)
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (n_cases, n_clusters)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
    return kmeans


def _get_model_centres(data, distance, average_params=None, distance_params=None):
    """Get the centres of a model."""
    model = TimeSeriesKMeans(
        averaging_method="ba",
        random_state=1,
        n_init=2,
        n_clusters=2,
        init="random",
        distance=distance,
        average_params=average_params,
        distance_params=distance_params,
    )
    model.fit(data)
    return model.cluster_centers_


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


def test_k_means_expected():
    """Test implementation of Kmeans."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmeans = TimeSeriesKMeans(
        averaging_method="mean",
        random_state=1,
        n_init=2,
        n_clusters=2,
        init="kmeans++",
        distance="euclidean",
    )
    train_predict = kmeans.fit_predict(X_train)
    train_mean_score = metrics.rand_score(y_train, train_predict)

    test_mean_result = kmeans.predict(X_test)
    mean_score = metrics.rand_score(y_test, test_mean_result)
    proba = kmeans.predict_proba(X_test)

    assert np.array_equal(test_mean_result, k_means_expected_results["mean_result"])
    assert mean_score == k_means_expected_results["mean_test_score"]
    assert train_mean_score == k_means_expected_results["mean_train_score"]
    assert kmeans.n_iter_ == k_means_expected_results["mean_iters"]
    assert np.array_equal(kmeans.labels_, k_means_expected_results["mean_labels"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (40, 2)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


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
        init="kmeans++",
        distance="dtw",
    )
    train_predict = kmeans.fit_predict(X_train[0:num_test_values])
    train_mean_score = metrics.rand_score(y_train[0:num_test_values], train_predict)

    test_mean_result = kmeans.predict(X_test[0:num_test_values])
    mean_score = metrics.rand_score(y_test[0:num_test_values], test_mean_result)
    proba = kmeans.predict_proba(X_test[0:num_test_values])

    assert np.array_equal(test_mean_result, k_means_expected_results["dba_result"])
    assert mean_score == k_means_expected_results["dba_test_score"]
    assert train_mean_score == k_means_expected_results["dba_train_score"]
    assert kmeans.n_iter_ == k_means_expected_results["dba_iters"]
    assert np.array_equal(kmeans.labels_, k_means_expected_results["dba_labels"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (5, 2)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


@pytest.mark.parametrize("distance", TEST_DISTANCE_WITH_CUSTOM_DISTANCE)
def test_k_mean_distances(distance):
    """Test that all distances work in k-mean."""
    dist, params = distance

    for key in params:
        curr_params = {
            "max_iter": 10,
            "averaging_method": "mean",
            "random_state": 1,
            "n_init": 1,
            "n_clusters": 3,
            "init": "kmeans++",
            "distance": dist,
            "distance_params": {key: params[key]},
        }
        # Univariate test
        with_param_kmeans = _run_kmeans_test(
            kmeans_params=curr_params, n_cases=40, n_channels=1, n_timepoints=10
        )

        # Multivariate test
        _run_kmeans_test(
            kmeans_params=curr_params, n_cases=40, n_channels=3, n_timepoints=10
        )

        if dist in ELASTIC_DISTANCES:
            curr_params["distance_params"] = {"window": 0.1}
        else:
            del curr_params["distance_params"]

        # Shift scale with random values doesn't change the centers (1 which is the
        # default is the best shift already)
        if dist == "shift_scale":
            continue

        default_param_kmeans = _run_kmeans_test(
            kmeans_params=curr_params, n_cases=40, n_channels=1, n_timepoints=10
        )

        # Test parameters passed through kmeans
        assert not np.array_equal(
            with_param_kmeans.cluster_centers_, default_param_kmeans.cluster_centers_
        )


@pytest.mark.parametrize("distance", TEST_DISTANCES_WITH_FULL_ALIGNMENT_PATH)
@pytest.mark.parametrize("averaging_method", ["subgradient", "petitjean", "kasba"])
def test_k_mean_ba(distance, averaging_method):
    """Test that all distances work in k-mean."""
    dist, params = distance

    for key in params:
        # Univariate test
        curr_params = {
            "max_iter": 10,
            "averaging_method": "ba",
            "random_state": 1,
            "n_init": 1,
            "n_clusters": 4,
            "init": "kmeans++",
            "distance": dist,
            "distance_params": {key: params[key]},
            "average_params": {
                "method": averaging_method,
                "max_iters": 4,
                "init_barycenter": "random",
            },
        }
        with_param_kmeans = _run_kmeans_test(
            kmeans_params=curr_params, n_cases=40, n_channels=1, n_timepoints=10
        )

        # Multivariate test
        _run_kmeans_test(
            kmeans_params=curr_params, n_cases=40, n_channels=3, n_timepoints=10
        )

        curr_params["distance_params"] = {"window": 0.1}

        default_param_kmeans = _run_kmeans_test(
            kmeans_params=curr_params, n_cases=40, n_channels=1, n_timepoints=10
        )

        # Test parameters passed through kmeans
        assert not np.array_equal(
            with_param_kmeans.cluster_centers_, default_param_kmeans.cluster_centers_
        )


@pytest.mark.parametrize("distance", TEST_DISTANCE_WITH_CUSTOM_DISTANCE)
@pytest.mark.parametrize("init", ["random", "kmeans++", "first", "ndarray"])
def test_k_mean_init(distance, init):
    """Test implementation of Kmeans."""
    distance, params = distance

    # Only kmeans++ needs test with different distances
    if init != "kmeans++" and distance != "euclidean":
        return

    n_cases = 10
    n_timepoints = 10
    n_clusters = 4

    # Univariate test
    X_train_uni = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=1,
        n_timepoints=n_timepoints,
        random_state=1,
        return_y=False,
    )

    kmeans = TimeSeriesKMeans(
        random_state=1,
        averaging_method="mean",
        init=init,
        distance=distance,
        n_clusters=n_clusters,
    )

    if init == "ndarray":
        kmeans.init = make_example_3d_numpy(
            n_cases=n_clusters,
            n_channels=1,
            n_timepoints=n_timepoints,
            random_state=1,
            return_y=False,
        )

    kmeans._check_params(X_train_uni)
    if isinstance(kmeans._init, Callable):
        uni_init_vals = kmeans._init(X_train_uni)
    else:
        uni_init_vals = kmeans._init

    check_value_in_every_cluster(n_clusters, uni_init_vals)

    # Multivariate test
    X_train_multi = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=3,
        n_timepoints=n_timepoints,
        random_state=1,
        return_y=False,
    )

    kmeans = TimeSeriesKMeans(
        random_state=1,
        averaging_method="mean",
        init=init,
        distance=distance,
        n_clusters=n_clusters,
    )

    if init == "ndarray":
        kmeans.init = make_example_3d_numpy(
            n_cases=n_clusters,
            n_channels=3,
            n_timepoints=n_timepoints,
            random_state=1,
            return_y=False,
        )

    kmeans._check_params(X_train_multi)

    if isinstance(kmeans._init, Callable):
        multi_init_vals = kmeans._init(X_train_multi)
    else:
        multi_init_vals = kmeans._init

    check_value_in_every_cluster(n_clusters, multi_init_vals)


def test_empty_cluster():
    """Test empty cluster handling."""
    first = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    second = np.array([[4, 5, 6], [7, 8, 9], [11, 12, 13]])
    third = np.array([[24, 25, 26], [27, 28, 29], [30, 31, 32]])
    forth = np.array([[14, 15, 16], [17, 18, 19], [20, 21, 22]])

    # Test where two swap must happen to avoid empty clusters
    empty_cluster = np.array([[100, 100, 100], [100, 100, 100], [100, 100, 100]])
    init_centres = np.array([first, empty_cluster, empty_cluster])

    kmeans = TimeSeriesKMeans(
        random_state=1,
        n_init=1,
        max_iter=5,
        init=init_centres,
        distance="euclidean",
        averaging_method="mean",
        n_clusters=3,
    )

    kmeans.fit(np.array([first, second, third, forth]))

    assert not np.array_equal(kmeans.cluster_centers_, init_centres)
    assert np.unique(kmeans.labels_).size == 3

    # Test that if a duplicate center would be created the algorithm
    init_centres = np.array([first, first, first])

    kmeans = TimeSeriesKMeans(
        random_state=1,
        n_init=1,
        max_iter=5,
        init=init_centres,
        distance="euclidean",
        averaging_method="mean",
        n_clusters=3,
    )

    kmeans.fit(np.array([first, second, third]))

    assert not np.array_equal(kmeans.cluster_centers_, init_centres)
    assert np.unique(kmeans.labels_).size == 3

    # Test duplicate data in dataset
    init_centres = np.array([first, empty_cluster])
    kmeans = TimeSeriesKMeans(
        random_state=1,
        n_init=1,
        max_iter=5,
        init=init_centres,
        distance="euclidean",
        averaging_method="mean",
        n_clusters=2,
    )

    kmeans.fit(np.array([first, first, first, first, second]))

    assert not np.array_equal(kmeans.cluster_centers_, init_centres)
    assert np.unique(kmeans.labels_).size == 2

    # Test impossible to have 3 different clusters
    init_centres = np.array([first, empty_cluster, empty_cluster])
    kmeans = TimeSeriesKMeans(
        random_state=1,
        n_init=1,
        max_iter=5,
        init=init_centres,
        distance="euclidean",
        averaging_method="mean",
        n_clusters=3,
    )

    with pytest.raises(ValueError):
        kmeans.fit(np.array([first, first, first, first, first]))


def test_invalid_params():
    """Test invalid parameters for k-mean."""
    uni_data = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=10, random_state=1, return_y=False
    )
    multi_data = make_example_3d_numpy(
        n_cases=10, n_channels=3, n_timepoints=10, random_state=1, return_y=False
    )
    # Init algorithm exceptions

    default_params = {
        "random_state": 1,
        "averaging_method": "mean",
        "distance": "euclidean",
        "n_init": 1,
        "max_iter": 5,
    }

    # Test invalid init string
    with pytest.raises(ValueError, match=r"invalid.*init|The value provided for init"):
        TimeSeriesKMeans(**default_params, n_clusters=2, init="not-a-valid-init").fit(
            uni_data
        )

    # Test different length init
    diff_len_uni_data = make_example_3d_numpy(
        n_cases=2, n_channels=1, n_timepoints=8, random_state=1, return_y=False
    )
    with pytest.raises(ValueError, match=r"invalid.*init|The value provided for init"):
        TimeSeriesKMeans(**default_params, n_clusters=2, init=diff_len_uni_data).fit(
            uni_data
        )

    # Test different dims init
    diff_len_multi_data = make_example_3d_numpy(
        n_cases=2, n_channels=5, n_timepoints=8, random_state=1, return_y=False
    )
    with pytest.raises(ValueError, match=r"invalid.*init|The value provided for init"):
        TimeSeriesKMeans(**default_params, n_clusters=2, init=diff_len_multi_data).fit(
            multi_data
        )

    # Test with invalid distance
    with pytest.raises(
        ValueError, match=r"Method must be one of the supported strings or a callable"
    ):
        TimeSeriesKMeans(
            n_clusters=2,
            init="first",
            averaging_method="mean",
            distance="not-a-real-dist",
            n_init=1,
        ).fit(uni_data)

    # Test with invalid distance for ba
    with pytest.raises(ValueError, match=r"Invalid distance passed for ba"):
        TimeSeriesKMeans(
            n_clusters=2,
            init="first",
            averaging_method="ba",
            distance="euclidean",
            n_init=1,
        ).fit(uni_data)

    # Test with invalid averaging method
    with pytest.raises(ValueError, match="averaging_method string is invalid"):
        TimeSeriesKMeans(
            n_clusters=2,
            init="first",
            averaging_method="no-real-average",
            distance="euclidean",
            n_init=1,
        ).fit(uni_data)

    with pytest.raises(ValueError, match=r"n_clusters .* cannot be larger"):
        TimeSeriesKMeans(n_clusters=20, init="first").fit(uni_data)

    # Test no clustering found
    with pytest.raises(
        ValueError, match=r"Unable to find a valid cluster configuration"
    ):
        TimeSeriesKMeans(
            n_clusters=2,
            n_init=1,
            max_iter=0,
            averaging_method="mean",
            init="random",
            distance="euclidean",
        ).fit(uni_data)


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("distance", TEST_DISTANCES_WITH_FULL_ALIGNMENT_PATH)
@pytest.mark.parametrize("averaging_method", ["subgradient", "petitjean", "kasba"])
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_k_means_ba_threaded(distance, averaging_method, n_jobs):
    """Test petitjean threaded functionality."""
    curr_params = {
        "max_iter": 10,
        "averaging_method": "ba",
        "random_state": 1,
        "n_init": 1,
        "n_jobs": n_jobs,
        "n_clusters": 4,
        "init": "kmeans++",
        "distance": distance[0],
        "average_params": {
            "method": averaging_method,
            "max_iters": 4,
            "init_barycenter": "random",
        },
    }
    _run_kmeans_test(
        kmeans_params=curr_params, n_cases=40, n_channels=1, n_timepoints=10
    )


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("distance", TEST_DISTANCE_WITH_PARAMS)
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_k_means_threaded(distance, n_jobs):
    """Test mean averaging threaded functionality."""
    curr_params = {
        "max_iter": 10,
        "averaging_method": "mean",
        "random_state": 1,
        "n_init": 1,
        "n_jobs": n_jobs,
        "n_clusters": 4,
        "init": "kmeans++",
        "distance": distance[0],
    }
    _run_kmeans_test(
        kmeans_params=curr_params, n_cases=40, n_channels=1, n_timepoints=10
    )
