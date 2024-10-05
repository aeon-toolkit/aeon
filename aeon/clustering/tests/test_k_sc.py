"""Test KSpectralCentroid clusterer."""

import numpy as np

from aeon.clustering import KSpectralCentroid
from aeon.datasets import load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy


def test_k_spectral_centroid_univariate():
    """Test KSpectralCentroid with univariate data."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    clusterer = KSpectralCentroid(n_clusters=2, random_state=1)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    assert clusterer.labels_.shape == (20,)
    assert len(clusterer.labels_) == 20
    assert len(set(clusterer.labels_)) == 2
    assert np.array_equal(clusterer.labels_, preds)
    assert clusterer.max_shift is None
    assert clusterer._distance_params["max_shift"] == data[0].shape[-1]
    assert clusterer._average_params["max_shift"] == data[0].shape[-1]

    assert clusterer.cluster_centers_.shape == (2, 1, 10)
    assert isinstance(clusterer.n_iter_, int)

    expected_labels = np.array(
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
    )
    assert np.array_equal(clusterer.labels_, expected_labels)
    assert np.array_equal(preds, expected_labels)


def test_k_spectral_centroid_multivariate():
    """Test KSpectralCentroid with multivariate data."""
    data = make_example_3d_numpy(20, 3, 10, return_y=False, random_state=1)
    clusterer = KSpectralCentroid(n_clusters=2, random_state=1)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    assert clusterer.labels_.shape == (20,)
    assert len(clusterer.labels_) == 20
    assert len(set(clusterer.labels_)) == 2
    assert np.array_equal(clusterer.labels_, preds)
    assert clusterer.max_shift is None
    assert clusterer._distance_params["max_shift"] == data[0].shape[-1]
    assert clusterer._average_params["max_shift"] == data[0].shape[-1]

    assert clusterer.cluster_centers_.shape == (2, 3, 10)
    assert isinstance(clusterer.n_iter_, int)

    expected_labels = np.array(
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    )
    assert np.array_equal(clusterer.labels_, expected_labels)
    assert np.array_equal(preds, expected_labels)


def test_k_spectral_centroid_with_max_shift():
    """Test KSpectralCentroid with different max_shift."""
    data, y_train = load_gunpoint(split="train")
    data = data[10:30]
    original_clusterer = KSpectralCentroid(n_clusters=2, random_state=1)
    original_clusterer.fit(data)
    clusterer = KSpectralCentroid(n_clusters=2, max_shift=2, random_state=42)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    assert clusterer.labels_.shape == (data.shape[0],)
    assert len(clusterer.labels_) == data.shape[0]
    assert len(set(clusterer.labels_)) == 2
    assert np.array_equal(clusterer.labels_, preds)

    assert not np.array_equal(clusterer.labels_, original_clusterer.labels_)
    assert clusterer.max_shift == 2
    assert clusterer._distance_params["max_shift"] == 2
    assert clusterer._average_params["max_shift"] == 2


def test_k_spectral_centroid_with_different_n_clusters():
    """Test KSpectralCentroid with different n_clusters."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    clusterer = KSpectralCentroid(n_clusters=4, random_state=1)
    clusterer.fit(data)
    assert clusterer.labels_.shape == (20,)
    assert len(clusterer.labels_) == 20
    assert len(set(clusterer.labels_)) == 4
    assert clusterer.max_shift is None
    assert clusterer._distance_params["max_shift"] == data[0].shape[-1]
    assert clusterer._average_params["max_shift"] == data[0].shape[-1]

    assert clusterer.cluster_centers_.shape == (4, 1, 10)
    assert isinstance(clusterer.n_iter_, int)
