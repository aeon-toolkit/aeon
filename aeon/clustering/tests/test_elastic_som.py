"""Tests for the ElasticSOM clustering algorithm."""

from aeon.clustering import ElasticSOM
from aeon.testing.data_generation import make_example_3d_numpy


def test_elastic_som_univariate():
    """Test ElasticSOM on a univariate dataset."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20, return_y=False)
    clst = ElasticSOM(n_clusters=2, random_state=1, num_iterations=10)
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (2, 1, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)


def test_elastic_som_multivariate():
    """Test ElasticSOM on a multivariate dataset."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=20, return_y=False)
    clst = ElasticSOM(n_clusters=2, random_state=1, num_iterations=10)
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (2, 2, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)
