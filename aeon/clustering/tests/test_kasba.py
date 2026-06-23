"""Test KASBA."""

import numpy as np
import pytest

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


def test_kasba_predict():
    """Test KASBA predict with a string distance."""
    X = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=False)
    kasba = KASBA(n_clusters=2, random_state=1, max_iter=2)
    kasba.fit(X)
    preds = kasba.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_kasba_verbose(capsys):
    """Test KASBA prints progress when verbose is True."""
    X = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=False)
    kasba = KASBA(n_clusters=2, random_state=1, max_iter=5, verbose=True)
    kasba.fit(X)
    out = capsys.readouterr().out
    assert "inertia" in out.lower() or "Iteration" in out


def test_kasba_n_clusters_too_large():
    """Test KASBA rejects n_clusters greater than the number of cases."""
    X = make_example_3d_numpy(5, 1, 10, random_state=1, return_y=False)
    with pytest.raises(ValueError, match="cannot be larger than"):
        KASBA(n_clusters=10).fit(X)


def test_kasba_handle_empty_cluster():
    """Test KASBA reassigns centres when a cluster becomes empty."""
    X = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=False)
    kasba = KASBA(n_clusters=3, distance="euclidean", random_state=1)
    kasba._distance_params = {}

    cluster_centers = X[:3].copy()
    # assign every case to cluster 0, leaving clusters 1 and 2 empty
    labels = np.zeros(20, dtype=int)
    distances_to_centers = np.ones(20)

    new_labels, new_centers, new_dists = kasba._handle_empty_cluster(
        X, cluster_centers, distances_to_centers, labels
    )
    assert set(new_labels.tolist()) == {0, 1, 2}
    assert new_centers.shape == (3, 1, 10)
