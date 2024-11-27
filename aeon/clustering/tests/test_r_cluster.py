"""Tests for time series R cluster."""

import numpy as np
import pytest

from aeon.clustering._r_cluster import RCluster
from aeon.datasets import load_basic_motions
from aeon.utils.validation._dependencies import _check_estimator_deps

expected_labels = [0, 2, 1, 2, 0]

expected_iters = 2

expected_results = [0, 0, 0, 0, 0]

@pytest.mark.skipif(
    not _check_estimator_deps( RCluster, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kernel_k_means():
    """Test implementation of R cluster."""
    max_train = 5

    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    r_cluster =  RCluster( n_clusters=2)
    r_cluster.fit(X_train[0:max_train])
    test_shape_result =  r_cluster.predict(X_test[0:max_train])


    assert np.array_equal(test_shape_result, expected_results)
    assert r_cluster.n_iter_ == expected_iters
    assert np.array_equal( r_cluster.labels_, expected_labels)
