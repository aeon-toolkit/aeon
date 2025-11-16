"""Tests for time series k-shape."""

import numpy as np
import pytest

from aeon.clustering._k_shape import TimeSeriesKShape
from aeon.datasets import load_basic_motions
from aeon.utils.validation._dependencies import _check_estimator_deps

expected_results = [2, 2, 2, 0, 0]

inertia = 0.5645477840468736

expected_iters = 2

expected_labels = [0, 2, 1, 1, 1]


@pytest.mark.skipif(
    not _check_estimator_deps(TimeSeriesKShape, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kshapes():
    """Test implementation of Kshapes."""
    max_train = 5

    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kshapes = TimeSeriesKShape(random_state=1, n_clusters=3)
    kshapes.fit(X_train[0:max_train])
    test_shape_result = kshapes.predict(X_test[0:max_train])
    proba = kshapes.predict_proba(X_test[0:max_train])
    assert np.array_equal(test_shape_result, expected_results)
    assert kshapes.n_iter_ == expected_iters
    assert np.array_equal(kshapes.labels_, expected_labels)
    assert proba.shape == (max_train, 3)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
