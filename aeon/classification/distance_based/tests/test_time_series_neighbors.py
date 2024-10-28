"""Tests for KNeighborsTimeSeriesClassifier."""

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
    # load arrowhead data for unit tests
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    knn = KNeighborsTimeSeriesClassifier(
        distance=distance_key,
    )
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
