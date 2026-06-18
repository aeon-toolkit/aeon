"""Tests for time series k-medoids."""

import numpy as np
import pytest
from sklearn import metrics

from aeon.clustering._clara import TimeSeriesCLARA
from aeon.datasets import load_basic_motions, load_gunpoint


def test_clara_uni():
    """Test implementation of CLARA."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    clara = TimeSeriesCLARA(
        random_state=1,
        n_samples=10,
        n_init=2,
        max_iter=5,
        init="first",
        distance="euclidean",
        n_clusters=2,
    )
    train_medoids_result = clara.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clara.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clara.predict_proba(X_test)
    assert np.array_equal(
        test_medoids_result,
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    )
    assert np.array_equal(
        train_medoids_result,
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    )
    assert test_score == 0.5210526315789473
    assert train_score == 0.5578947368421052
    assert np.isclose(clara.inertia_, 74.72628097332178)
    assert clara.n_iter_ == 3
    assert np.array_equal(
        clara.labels_, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]
    )
    assert isinstance(clara.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_clara_multi():
    """Test implementation of CLARA."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    clara = TimeSeriesCLARA(
        random_state=1,
        n_samples=10,
        n_init=2,
        max_iter=5,
        init="first",
        distance="euclidean",
        n_clusters=2,
    )
    train_medoids_result = clara.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clara.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clara.predict_proba(X_test)
    assert np.array_equal(
        test_medoids_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert np.array_equal(
        train_medoids_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    )
    assert test_score == 0.4789473684210526
    assert train_score == 0.5578947368421052
    assert np.isclose(clara.inertia_, 1675.1873875545991)
    assert clara.n_iter_ == 3
    assert np.array_equal(
        clara.labels_, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
    )
    assert isinstance(clara.cluster_centers_, np.ndarray)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_clara_samples_from_all_cases():
    """Test CLARA samples from all cases, not only the first n_samples."""
    X = np.arange(20).reshape(20, 1, 1).astype(float)

    clara = TimeSeriesCLARA(
        n_clusters=10,
        init="first",
        n_samples=10,
        n_sampling_iters=1,
        n_init=1,
        max_iter=1,
        distance="euclidean",
        random_state=1,
    )

    clara.fit(X)

    assert np.max(clara.cluster_centers_) >= clara.n_samples


def test_clara_init_indices_remapped():
    """Test CLARA correctly remaps init indices when subsampling (issue #3423)."""
    X = np.arange(40).reshape(20, 2, 1).astype(float)

    # Use indices that would be out of bounds in a small sample
    clara = TimeSeriesCLARA(
        n_clusters=2,
        init=np.array([5, 15]),
        n_samples=4,
        n_sampling_iters=1,
        n_init=1,
        max_iter=2,
        distance="euclidean",
        random_state=1,
    )

    clara.fit(X)

    # Check that it fitted successfully and produced labels
    assert clara.labels_ is not None
    assert len(clara.labels_) == 20
    assert clara.cluster_centers_ is not None


@pytest.mark.parametrize(
    "init, match",
    [
        (np.array([-1, 0]), "Values must be in the range"),
        (np.array([0, 30]), "Values must be in the range"),
        (np.array([0.5, 1.0]), "Expected an array of integers"),
    ],
)
def test_clara_init_ndarray_validation(init, match):
    """Test CLARA raises ValueError for invalid ndarray init values."""
    X = np.arange(40).reshape(20, 2, 1).astype(float)

    clara = TimeSeriesCLARA(
        n_clusters=2,
        init=init,
        n_samples=5,
        n_sampling_iters=1,
        n_init=1,
        max_iter=1,
        distance="euclidean",
        random_state=1,
    )

    with pytest.raises(ValueError, match=match):
        clara.fit(X)
