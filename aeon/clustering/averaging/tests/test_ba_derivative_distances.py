"""Tests of the barycenter averaging with the derivative based distances."""

import numpy as np
import pytest

from aeon.clustering import TimeSeriesKMeans
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.clustering.averaging._ba_utils import _get_alignment_path


@pytest.mark.parametrize("distance", ["ddtw", "wddtw"])
def test_alignment_path_covers_every_time_point(distance):
    """The path of a derivative distance is expressed on the original series."""
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(1, 20))
    center = rng.normal(size=(1, 24))

    path, _ = _get_alignment_path(center, ts, distance=distance)

    assert path[0] == (0, 0)
    assert path[-1] == (19, 23)
    assert {j for j, _ in path} == set(range(20))
    assert {k for _, k in path} == set(range(24))


@pytest.mark.parametrize("distance", ["ddtw", "wddtw"])
@pytest.mark.parametrize("method", ["petitjean", "subgradient", "kasba"])
def test_barycenter_of_derivative_distances_is_finite(distance, method):
    """Every time point of the barycenter is updated, none is left as NaN."""
    X = np.random.default_rng(0).normal(size=(8, 1, 24))

    barycenter = elastic_barycenter_average(
        X, distance=distance, method=method, max_iters=5
    )

    assert barycenter.shape == (1, 24)
    assert np.isfinite(barycenter).all()
    if method != "kasba":
        assert not np.allclose(barycenter[:, -2:], X.mean(axis=0)[:, -2:])


@pytest.mark.parametrize("distance", ["ddtw", "wddtw"])
def test_kmeans_with_derivative_distance_and_ba_fits(distance):
    """Clustering with ba averaging finds non empty clusters and finite centres."""
    X = np.random.default_rng(1).normal(size=(12, 1, 24))

    clusterer = TimeSeriesKMeans(
        n_clusters=2,
        distance=distance,
        averaging_method="ba",
        max_iter=3,
        n_init=1,
        random_state=0,
    ).fit(X)

    assert np.isfinite(clusterer.cluster_centers_).all()
    assert len(np.unique(clusterer.labels_)) == 2
