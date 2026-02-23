"""Test KASBA."""

import numpy as np

from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
from aeon.clustering import KASBA
from aeon.testing.data_generation import make_example_3d_numpy


def test_univariate_kasba():
    """Test KASBA on univariate data."""
    X, y = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=True)

    kasba = KASBA(n_clusters=len(np.unique(y)), random_state=1)

    kasba.fit(X)
    score = clustering_accuracy_score(y, kasba.labels_)
    assert score == 0.95


def test_multivariate_kasba():
    """Test KASBA on multivariate data."""
    X, y = make_example_3d_numpy(20, 3, 10, random_state=1, return_y=True)

    kasba = KASBA(n_clusters=len(np.unique(y)), random_state=1)

    kasba.fit(X)
    score = clustering_accuracy_score(y, kasba.labels_)
    assert score == 0.55
