"""Test TimeSeriesAgglomerative."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from aeon.clustering._agglomerativeClustering import TimeSeriesAgglomerative
from aeon.testing.data_generation import make_example_3d_numpy

LINKAGES = ["single", "complete", "average"]


def _sklearn_hc(X, n_clusters, linkage):
    """Run scikit-learn agglomerative clustering on flattened time series."""
    X_flat = X.reshape(X.shape[0], -1)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage=linkage,
    )
    return model.fit_predict(X_flat)


def _aeon_hc(X, n_clusters, linkage):
    """Run TimeSeriesAgglomerative using Euclidean distance."""
    model = TimeSeriesAgglomerative(
        n_clusters=n_clusters,
        distance="euclidean",
        linkage=linkage,
    )
    return model.fit_predict(X)


def _check_hc(X, y, linkage):
    """Check that sklearn and aeon return equivalent clusterings."""
    n_clusters = len(np.unique(y))
    sklearn_labels = _sklearn_hc(X, n_clusters, linkage)
    aeon_labels = _aeon_hc(X, n_clusters, linkage)
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
