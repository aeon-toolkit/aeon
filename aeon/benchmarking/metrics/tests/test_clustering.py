"""Tests for performance metric functions."""

import numpy as np

from aeon.benchmarking.metrics.clustering import clustering_accuracy_score


def test_clustering_accuracy():
    """Test clustering accuracy with random labels and clusters."""
    labels = np.random.randint(0, 3, 10)
    clusters = np.random.randint(0, 3, 10)
    cl_acc = clustering_accuracy_score(labels, clusters)

    assert isinstance(cl_acc, float)
    assert 0 <= cl_acc <= 1
