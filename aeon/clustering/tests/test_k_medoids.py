"""Tests for time series k-medoids."""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning

from aeon.clustering._cluster_initialisation import _CENTRE_INITIALISER_INDEXES
from aeon.clustering._k_medoids import TimeSeriesKMedoids
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.distances import euclidean_distance
from aeon.testing.data_generation import make_example_3d_numpy


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

    # Test duplicate indices
    with pytest.raises(ValueError, match="unique indices"):
        kmedoids = TimeSeriesKMedoids(
            n_clusters=num_clusters,
            init=np.array([0, 1, 1]),
            random_state=1,
        )
        kmedoids.fit(X_train)


def test_kmedoids_build_init():
    """Test k-medoids with the greedy 'build' initialisation strategy."""
    X = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    kmedoids = TimeSeriesKMedoids(
        n_clusters=3,
        init="build",
        distance="euclidean",
        method="pam",
        n_init=1,
        max_iter=5,
        random_state=1,
    )
    kmedoids.fit(X)
    assert kmedoids.cluster_centers_.shape == (3, 1, 8)
    assert len(set(kmedoids.labels_)) <= 3


def test_kmedoids_build_warns_on_n_init():
    """Test 'build' init warns when n_init is greater than 1."""
    X = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    kmedoids = TimeSeriesKMedoids(
        n_clusters=2,
        init="build",
        distance="euclidean",
        method="pam",
        n_init=5,
        max_iter=3,
        random_state=1,
    )
    with pytest.warns(UserWarning, match="n_init will be set to 1"):
        kmedoids.fit(X)


def test_kmedoids_invalid_method():
    """Test k-medoids rejects an unknown method."""
    X = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    with pytest.raises(ValueError, match="is not supported"):
        TimeSeriesKMedoids(n_clusters=2, method="invalid", distance="euclidean").fit(X)


def test_kmedoids_n_clusters_too_large():
    """Test k-medoids rejects n_clusters greater than the number of cases."""
    X = make_example_3d_numpy(5, 1, 8, return_y=False, random_state=1)
    with pytest.raises(ValueError, match="cannot be larger than"):
        TimeSeriesKMedoids(n_clusters=10, init="first", distance="euclidean").fit(X)


def test_kmedoids_callable_distance():
    """Test k-medoids with a callable distance function."""
    X = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    kmedoids = TimeSeriesKMedoids(
        n_clusters=2,
        distance=euclidean_distance,
        method="pam",
        init="first",
        n_init=1,
        max_iter=3,
        random_state=1,
    )
    kmedoids.fit(X)
    preds = kmedoids.predict(X)
    assert preds.shape == (10,)


@pytest.mark.parametrize("method", ["pam", "alternate"])
def test_kmedoids_ndarray_init(method):
    """Test k-medoids accepts an explicit array of medoid indices."""
    X = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    kmedoids = TimeSeriesKMedoids(
        n_clusters=2,
        init=np.array([0, 5]),
        distance="euclidean",
        method=method,
        n_init=1,
        max_iter=3,
        random_state=1,
    )
    kmedoids.fit(X)
    assert kmedoids.cluster_centers_.shape == (2, 1, 8)


@pytest.mark.parametrize("method", ["pam", "alternate"])
def test_kmedoids_verbose(method, capsys):
    """Test k-medoids prints progress when verbose is True."""
    X = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    kmedoids = TimeSeriesKMedoids(
        n_clusters=2,
        distance="euclidean",
        method=method,
        init="first",
        n_init=1,
        max_iter=10,
        verbose=True,
        random_state=1,
    )
    kmedoids.fit(X)
    out = capsys.readouterr().out
    assert "Iteration" in out or "Converged" in out


def test_kmedoids_max_iter_warning():
    """Test k-medoids warns when it fails to converge in max_iter."""
    X = make_example_3d_numpy(20, 1, 12, return_y=False, random_state=1)
    kmedoids = TimeSeriesKMedoids(
        n_clusters=3,
        distance="euclidean",
        method="pam",
        init="first",
        n_init=1,
        max_iter=1,
        random_state=1,
    )
    with pytest.warns(ConvergenceWarning):
        kmedoids.fit(X)


def test_kmedoids_get_test_params():
    """Test the testing parameter set is valid."""
    params = TimeSeriesKMedoids._get_test_params()
    assert params["n_clusters"] == 2
    assert TimeSeriesKMedoids(**params).fit(
        make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    )
