"""Test TimeSeriesAgglomerative."""

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from aeon.clustering import TimeSeriesAgglomerative
from aeon.testing.data_generation import make_example_3d_numpy

LINKAGES = ["single", "complete", "average"]


def _sklearn_hc(X, n_clusters, linkage):
    # Flatten the time series so sklearn uses standard Euclidean distance
    # on the same values that aeon receives as 3D time series.
    X_flat = X.reshape(X.shape[0], -1)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage=linkage,
    )
    return model.fit_predict(X_flat)


def _aeon_hc(X, n_clusters, linkage):
    # TimeSeriesAgglomerative computes the Euclidean pairwise distance matrix
    # through aeon, then passes it to sklearn as a precomputed matrix.
    model = TimeSeriesAgglomerative(
        n_clusters=n_clusters,
        distance="euclidean",
        linkage=linkage,
    )
    return model.fit_predict(X)


def _check_hc(X, y, linkage):
    n_clusters = len(np.unique(y))
    sklearn_labels = _sklearn_hc(X, n_clusters, linkage)
    aeon_labels = _aeon_hc(X, n_clusters, linkage)
    # ARI compares the partition structure, ignoring arbitrary label names.
    ari = adjusted_rand_score(sklearn_labels, aeon_labels)

    assert np.isclose(ari, 1.0)
    assert aeon_labels.shape == (X.shape[0],)
    assert len(np.unique(aeon_labels)) <= n_clusters


def test_univariate_hc():
    """Test TimeSeriesAgglomerative on univariate data."""
    X, y = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=True)
    for linkage in LINKAGES:
        _check_hc(X, y, linkage)


def test_multivariate_hc():
    """Test TimeSeriesAgglomerative on multivariate data."""
    X, y = make_example_3d_numpy(20, 3, 10, random_state=1, return_y=True)
    for linkage in LINKAGES:
        _check_hc(X, y, linkage)


@pytest.mark.parametrize("linkage", ["ward", "bad"])
def test_invalid_linkage_raises(linkage):
    """Test invalid linkage options raise an error."""
    X = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=False)
    # Ward requires raw Euclidean feature vectors, so it is not valid when
    # TimeSeriesAgglomerative passes a precomputed distance matrix to sklearn.
    model = TimeSeriesAgglomerative(
        n_clusters=2,
        distance="euclidean",
        linkage=linkage,
    )

    with pytest.raises(ValueError, match="linkage|Ward"):
        model.fit(X)


@pytest.mark.parametrize(
    "params",
    [
        {"n_clusters": None, "distance_threshold": None},
        {"n_clusters": 2, "distance_threshold": 1.0},
        {"n_clusters": 21},
    ],
)
def test_invalid_cluster_settings_raise(params):
    """Test invalid cluster settings raise an error."""
    X = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=False)
    # These settings cover mutually exclusive stopping criteria and too many
    # clusters for the number of input cases.
    model = TimeSeriesAgglomerative(distance="euclidean", **params)

    with pytest.raises(ValueError):
        model.fit(X)


def test_asymmetric_distance_matrix_raises(monkeypatch):
    """Test asymmetric pairwise distances raise an error."""

    def asymmetric_distance(*args, **kwargs):
        # This mimics a broken or inappropriate distance implementation.
        return np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        )

    # Patch the imported function in _agglomerative so _pairwise_distance receives
    # this asymmetric matrix instead of calling aeon.distances.pairwise_distance.
    monkeypatch.setattr(
        "aeon.clustering._agglomerative.pairwise_distance",
        asymmetric_distance,
    )
    X = make_example_3d_numpy(3, 1, 10, random_state=1, return_y=False)
    model = TimeSeriesAgglomerative(n_clusters=2, distance="euclidean")

    with pytest.raises(ValueError, match="symmetric"):
        model.fit(X)


def test_non_finite_distance_matrix_raises(monkeypatch):
    """Test non-finite pairwise distances raise an error."""

    def non_finite_distance(*args, **kwargs):
        # This checks the wrapper validation before the matrix is passed to sklearn.
        return np.array(
            [
                [0.0, np.inf, 2.0],
                [np.inf, 0.0, 4.0],
                [2.0, 4.0, 0.0],
            ]
        )

    monkeypatch.setattr(
        "aeon.clustering._agglomerative.pairwise_distance",
        non_finite_distance,
    )
    X = make_example_3d_numpy(3, 1, 10, random_state=1, return_y=False)
    model = TimeSeriesAgglomerative(n_clusters=2, distance="euclidean")

    with pytest.raises(ValueError, match="NaN|infinite"):
        model.fit(X)
