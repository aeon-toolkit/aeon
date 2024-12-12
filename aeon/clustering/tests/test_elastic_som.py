"""Tests for the ElasticSOM clustering algorithm."""

import numpy as np
import pytest

from aeon.clustering import ElasticSOM
from aeon.distances import dtw_distance, msm_alignment_path
from aeon.distances._distance import ELASTIC_DISTANCES
from aeon.testing.data_generation import make_example_3d_numpy


def test_elastic_som_univariate():
    """Test ElasticSOM on a univariate dataset."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=20, return_y=False, random_state=1
    )
    clst = ElasticSOM(n_clusters=3, random_state=1, num_iterations=10)
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 1, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)


def test_elastic_som_multivariate():
    """Test ElasticSOM on a multivariate dataset."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    clst = ElasticSOM(n_clusters=3, random_state=1, num_iterations=10)
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)


def test_elastic_som_init():
    """Test ElasticSOM with a custom initialization."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    labels = []
    for init in ["random", "kmeans++", "first"]:
        clst = ElasticSOM(n_clusters=3, init=init, random_state=1, num_iterations=10)
        clst.fit(X)
        assert clst.labels_.shape == (10,)
        assert clst.cluster_centers_.shape == (3, 5, 20)
        labels.append(clst.labels_)

        preds = clst.predict(X)
        assert preds.shape == (10,)

    # Check that the labels are different
    assert not np.array_equal(labels[0], labels[1])
    assert not np.array_equal(labels[0], labels[2])
    assert not np.array_equal(labels[1], labels[2])
    # Test invalid init
    with pytest.raises(ValueError):
        clst = ElasticSOM(
            n_clusters=3, init="invalid", random_state=1, num_iterations=10
        )
        clst.fit(X)

    # Test custom ndarray init
    clst = ElasticSOM(n_clusters=3, init=X[:3], random_state=1, num_iterations=10)
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    # Last labels is for "first" init
    assert np.array_equal(clst.labels_, labels[-1])

    preds = clst.predict(X)
    assert preds.shape == (10,)

    # Test more ndarrays than clusters
    with pytest.raises(ValueError):
        clst = ElasticSOM(n_clusters=3, init=X[:4], random_state=1, num_iterations=10)
        clst.fit(X)


def test_elastic_som_decay_function():
    """Test ElasticSOM with a custom decay function."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    labels = []
    for decay_function in [
        "asymptotic_decay",
        "inverse_decay_to_zero",
        "linear_decay_to_zero",
    ]:
        clst = ElasticSOM(
            n_clusters=3,
            decay_function=decay_function,
            random_state=1,
            num_iterations=10,
        )
        clst.fit(X)
        assert clst.labels_.shape == (10,)
        assert clst.cluster_centers_.shape == (3, 5, 20)

        labels.append(clst.labels_)

        preds = clst.predict(X)
        assert preds.shape == (10,)

    # Check that the labels are different
    assert not np.array_equal(labels[0], labels[1])
    assert not np.array_equal(labels[0], labels[2])
    assert not np.array_equal(labels[1], labels[2])

    def custom_decay_function(learning_rate, current_iteration, max_iter):
        return learning_rate * current_iteration / max_iter

    # Test custom decay function
    clst = ElasticSOM(
        n_clusters=3,
        decay_function=custom_decay_function,
        random_state=1,
        num_iterations=10,
    )

    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)

    # Test invalid decay function
    with pytest.raises(ValueError):
        clst = ElasticSOM(
            n_clusters=3, decay_function="invalid", random_state=1, num_iterations=10
        )
        clst.fit(X)


def test_elastic_som_signama_decay_function():
    """Test ElasticSOM with a custom sigma decay function."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    for sigma_decay_function in [
        "asymptotic_decay",
        "inverse_decay_to_one",
        "linear_decay_to_one",
    ]:
        clst = ElasticSOM(
            n_clusters=3,
            sigma_decay_function=sigma_decay_function,
            random_state=1,
            num_iterations=10,
        )
        clst.fit(X)
        assert clst.labels_.shape == (10,)
        assert clst.cluster_centers_.shape == (3, 5, 20)

        preds = clst.predict(X)
        assert preds.shape == (10,)

    def custom_sigma_decay_function(sigma, current_iteration, max_iter):
        return sigma * (1 - (current_iteration / max_iter))

    # Test custom sigma decay function
    clst = ElasticSOM(
        n_clusters=3,
        sigma_decay_function=custom_sigma_decay_function,
        random_state=1,
        num_iterations=10,
    )

    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)

    # Test invalid sigma decay function
    with pytest.raises(ValueError):
        clst = ElasticSOM(
            n_clusters=3,
            sigma_decay_function="invalid",
            random_state=1,
            num_iterations=10,
        )
        clst.fit(X)


def test_elastic_som_neighborhood_function():
    """Test ElasticSOM with a custom neighborhood function."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    labels = []
    for neighborhood_function in ["gaussian", "mexican_hat"]:
        clst = ElasticSOM(
            n_clusters=3,
            neighborhood_function=neighborhood_function,
            random_state=1,
            num_iterations=10,
        )
        clst.fit(X)
        assert clst.labels_.shape == (10,)
        assert clst.cluster_centers_.shape == (3, 5, 20)

        labels.append(clst.labels_)

        preds = clst.predict(X)
        assert preds.shape == (10,)

    # Check that the labels are different
    assert not np.array_equal(labels[0], labels[1])

    # Test custom neighborhood function
    def custom_neighborhood_function(neuron_position, c, sigma):
        return np.exp(-np.power(neuron_position - neuron_position[c], 2) / 2)

    clst = ElasticSOM(
        n_clusters=3,
        neighborhood_function=custom_neighborhood_function,
        random_state=1,
        num_iterations=10,
    )

    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)

    # Test invalid neighborhood function
    with pytest.raises(ValueError):
        clst = ElasticSOM(
            n_clusters=3,
            neighborhood_function="invalid",
            random_state=1,
            num_iterations=10,
        )
        clst.fit(X)


@pytest.mark.parametrize("dist", ELASTIC_DISTANCES)
def test_elastic_som_distances(dist):
    """Test ElasticSOM distances."""
    if "distance" not in dist:
        return

    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    clst = ElasticSOM(
        n_clusters=3, distance=dist["name"], random_state=1, num_iterations=10
    )
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)


def test_elastic_som_custom_distance_and_params():
    """Test ElasticSOM custom distance and params."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )
    clst = ElasticSOM(n_clusters=3, distance="dtw", random_state=1, num_iterations=10)
    clst.fit(X)

    dtw_labels = clst.labels_

    clst = ElasticSOM(
        n_clusters=3,
        distance="dtw",
        distance_params={"window": 0.2},
        random_state=1,
        num_iterations=10,
    )

    clst.fit(X)
    dtw_window_labels = clst.labels_
    assert not np.array_equal(dtw_labels, dtw_window_labels)

    def custom_distance(x, y, window=0.2):
        return dtw_distance(x, y, window=window)

    clst = ElasticSOM(
        n_clusters=3,
        distance=custom_distance,
        distance_params={"window": 0.2},
        random_state=1,
        num_iterations=10,
    )

    clst.fit(X)
    custom_window_labels = clst.labels_
    assert np.array_equal(dtw_window_labels, custom_window_labels)

    preds = clst.predict(X)
    assert preds.shape == (10,)


def test_elastic_som_custom_alignment_path():
    """Test ElasticSOM custom alignment path."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=5, n_timepoints=20, return_y=False, random_state=1
    )

    def custom_alignment_path(x, y):
        return msm_alignment_path(x, y)

    clst = ElasticSOM(
        n_clusters=3,
        distance="dtw",
        custom_alignment_path=custom_alignment_path,
        random_state=1,
        num_iterations=10,
    )
    clst.fit(X)
    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (3, 5, 20)

    preds = clst.predict(X)
    assert preds.shape == (10,)

    first_labels = clst.labels_

    clst = ElasticSOM(n_clusters=3, distance="dtw", random_state=1, num_iterations=10)

    clst.fit(X)
    second_labels = clst.labels_

    assert not np.array_equal(first_labels, second_labels)
