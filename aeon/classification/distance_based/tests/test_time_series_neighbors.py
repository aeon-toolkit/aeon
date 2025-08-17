"""Tests for KNeighborsTimeSeriesClassifier."""

import numpy as np
import pytest

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_unit_test
from aeon.distances import get_distance_function

distance_functions = [
    "euclidean",
    "dtw",
    "wdtw",
    "msm",
    "erp",
    "lcss",
    "edr",
]

# expected correct on test set using default parameters.
expected_correct = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 20,
    "erp": 19,
    "lcss": 12,
    "edr": 20,
}

# expected correct on test set using window params.
expected_correct_window = {
    "euclidean": 19,
    "dtw": 21,
    "wdtw": 21,
    "msm": 20,
    "erp": 19,
    "edr": 20,
    "lcss": 12,
}


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_on_unit_test(distance_key):
    """Test function for elastic knn, to be reinstated soon."""
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    knn = KNeighborsTimeSeriesClassifier(distance=distance_key)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct[distance_key]


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_bounding_matrix(distance_key):
    """Test knn with custom bounding parameters."""
    if distance_key == "euclidean" or distance_key == "squared":
        return
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    distance_callable = get_distance_function(distance_key)

    knn = KNeighborsTimeSeriesClassifier(
        distance=distance_callable, distance_params={"window": 0.5}
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct_window[distance_key]


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_kneighbors(distance_key):
    """Test knn kneighbors with comprehensive validation."""
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    knn = KNeighborsTimeSeriesClassifier(distance=distance_key)
    knn.fit(X_train, y_train)

    # Test basic kneighbors functionality
    dists, ind = knn.kneighbors(X_test, n_neighbors=3)
    assert isinstance(dists, np.ndarray)
    assert isinstance(ind, np.ndarray)
    assert dists.shape == (X_test.shape[0], 3)
    assert ind.shape == (X_test.shape[0], 3)

    # Test that distances are non-negative
    assert np.all(dists >= 0)

    # Test that indices are within valid range
    assert np.all(ind >= 0)
    assert np.all(ind < len(X_train))

    # Test that distances are sorted (closest first)
    assert np.all(dists[:, 0] <= dists[:, 1])
    assert np.all(dists[:, 1] <= dists[:, 2])

    # Test using kneighbors results to make predictions manually
    # This validates that the kneighbors method returns correct neighbor indices
    indexes = ind[:, 0]
    classes, y = np.unique(y_train, return_inverse=True)
    pred = classes[y[indexes]]
    correct = 0
    for j in range(0, len(pred)):
        if pred[j] == y_test[j]:
            correct = correct + 1
    assert correct == expected_correct_window[distance_key]

    # Test kneighbors with different n_neighbors values
    dists_2, ind_2 = knn.kneighbors(X_test, n_neighbors=2)
    assert dists_2.shape == (X_test.shape[0], 2)
    assert ind_2.shape == (X_test.shape[0], 2)

    # Test kneighbors without returning distances
    ind_only = knn.kneighbors(X_test, n_neighbors=3, return_distance=False)
    assert isinstance(ind_only, np.ndarray)
    assert ind_only.shape == (X_test.shape[0], 3)
    # Should return same indices as when return_distance=True
    np.testing.assert_array_equal(ind_only, ind)

    # Test kneighbors on training data (should exclude self)
    train_dists, train_ind = knn.kneighbors(n_neighbors=2)
    assert train_dists.shape == (len(X_train), 2)
    assert train_ind.shape == (len(X_train), 2)
    # Each point should not be its own neighbor (diagonal should be excluded)
    for i in range(len(X_train)):
        assert i not in train_ind[i]
