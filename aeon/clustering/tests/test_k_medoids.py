"""Tests for time series k-medoids."""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn import metrics

from aeon.clustering._cluster_initialisation import _CENTRE_INITIALISER_INDEXES
from aeon.clustering._k_medoids import TimeSeriesKMedoids
from aeon.datasets import load_basic_motions, load_gunpoint


def test_kmedoids_uni():
    """Test implementation of Kmedoids."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    num_points = 10

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    _alternate_uni_medoids(X_train, y_train, X_test, y_test)
    _pam_uni_medoids(X_train, y_train, X_test, y_test)


def test_kmedoids_multi():
    """Test implementation of Kmedoids for multivariate."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 10

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    _alternate_multi_medoids(X_train, y_train, X_test, y_test)
    _pam_multi_medoids(X_train, y_train, X_test, y_test)


def _pam_uni_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init="first",
        distance="euclidean",
        method="pam",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [3, 5, 7, 3, 5, 5, 3, 1, 2, 5])
    assert np.array_equal(train_medoids_result, [0, 2, 2, 5, 4, 5, 6, 7, 1, 3])
    assert test_score == 0.5777777777777777
    assert train_score == 0.4222222222222222
    assert np.isclose(kmedoids.inertia_, 5.0087431726326646)
    assert kmedoids.n_iter_ == 3
    assert np.array_equal(kmedoids.labels_, [0, 2, 2, 5, 4, 5, 6, 7, 1, 3])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _alternate_uni_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        method="alternate",
        init="first",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [6, 1, 7, 6, 3, 5, 6, 6, 2, 5])
    assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 6, 6])
    assert test_score == 0.5333333333333333
    assert train_score == 0.4444444444444444
    assert np.isclose(kmedoids.inertia_, 13.942955492757196)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 6, 6])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _pam_multi_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init="first",
        distance="euclidean",
        method="pam",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    assert np.array_equal(test_medoids_result, [1, 4, 1, 0, 4, 3, 4, 3, 3, 4])
    assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 4, 6, 7, 4, 5])
    assert test_score == 0.2222222222222222
    assert train_score == 0.06666666666666667
    assert np.isclose(kmedoids.inertia_, 14.729547948813156)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 4, 6, 7, 4, 5])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def _alternate_multi_medoids(X_train, y_train, X_test, y_test):
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=2,
        max_iter=5,
        init="first",
        method="alternate",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)

    assert np.array_equal(test_medoids_result, [1, 5, 1, 0, 5, 3, 5, 5, 3, 4])
    assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 4, 4])
    assert test_score == 0.17777777777777778
    assert train_score == 0.06666666666666667
    assert np.isclose(kmedoids.inertia_, 20.66525401745263)
    assert kmedoids.n_iter_ == 2
    assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 4, 4])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def check_value_in_every_cluster(num_clusters, initial_medoids):
    """Check that every cluster has at least one value."""
    original_length = len(initial_medoids)
    assert original_length == num_clusters
    if isinstance(initial_medoids, np.ndarray):
        for i in range(len(initial_medoids)):
            curr = initial_medoids[i]
            for j in range(len(initial_medoids)):
                if i == j:
                    continue
                other = initial_medoids[j]
                assert not np.array_equal(curr, other)
    else:
        assert original_length == len(set(initial_medoids))


@pytest.mark.parametrize("init", list(_CENTRE_INITIALISER_INDEXES.keys()) + ["indexes"])
def test_medoids_init(init):
    """Test implementation of Kmedoids."""
    X_train, _ = load_gunpoint(split="train")
    X_train = X_train[:10]

    num_clusters = 3

    if init == "indexes":
        # Generate random indexes
        rng = np.random.RandomState(1)
        init = rng.choice(X_train.shape[0], num_clusters, replace=False)

    # Test initializer
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=1,
        max_iter=5,
        init=init,
        distance="euclidean",
        n_clusters=num_clusters,
    )
    kmedoids._check_params(X_train)
    if isinstance(kmedoids._init, Callable):
        medoids_result = kmedoids._init(X=X_train)
    else:
        medoids_result = kmedoids._init
    check_value_in_every_cluster(num_clusters, medoids_result)


def _get_model_centres(data, distance, method="pam", distance_params=None):
    """Get the centres of a model."""
    model = TimeSeriesKMedoids(
        random_state=1,
        method=method,
        n_init=2,
        n_clusters=2,
        init="random",
        distance=distance,
        distance_params=distance_params,
    )
    model.fit(data)
    return model.cluster_centers_


def test_custom_distance_params():
    """Test kmedoids custom distance parameters."""
    X_train, y_train = load_basic_motions(split="train")

    num_test_values = 10
    data = X_train[0:num_test_values]

    # Test passing distance param
    default_dist = _get_model_centres(data, distance="msm")
    custom_params_dist = _get_model_centres(
        data, distance="msm", distance_params={"window": 0.01}
    )
    assert not np.array_equal(default_dist, custom_params_dist)


def test_medoids_init_invalid():
    """Test implementation of Kmedoids with invalid init."""
    X_train, _ = load_gunpoint(split="train")
    X_train = X_train[:10]
    num_clusters = 3

    # Test float array
    with pytest.raises(ValueError, match="Expected an array of integers"):
        kmedoids = TimeSeriesKMedoids(
            n_clusters=num_clusters,
            init=np.array([0.5, 1.5, 2.5]),
            random_state=1,
        )
        kmedoids.fit(X_train)

    # Test out of bounds
    with pytest.raises(ValueError, match="Values must be in the range"):
        kmedoids = TimeSeriesKMedoids(
            n_clusters=num_clusters,
            init=np.array([0, 1, 100]),
            random_state=1,
        )
        kmedoids.fit(X_train)
