"""Tests for UShapeletClusterer."""

import numpy as np

from aeon.clustering._u_shapelet import UShapeletClusterer


def create_synthetic_data_three_clusters():
    """
    Create synthetic dataset of univariate time series that has three natural clusters.

    Cluster 0: Lower-valued time series.
    Cluster 1: Medium-valued time series.
    Cluster 2: Higher-valued time series.
    """
    cluster0 = [np.array([1, 2, 2, 3, 3, 2]) for _ in range(3)]
    cluster1 = [np.array([8, 8, 9, 9, 10, 9]) for _ in range(3)]
    cluster2 = [np.array([50, 50, 51, 51, 52, 52]) for _ in range(3)]

    return cluster0 + cluster1 + cluster2


def test_ushapelet_clusterer_two_clusters():
    """Test UShapeletClusterer with default n_clusters=2 on our synthetic data."""
    data = create_synthetic_data_three_clusters()

    # single-shapelet approach (2 clusters)
    ushapelet = UShapeletClusterer(
        subseq_length=3,
        sax_length=4,
        projections=5,
        lb=1,
        ub=4,
        random_state=18,
    )
    ushapelet.fit(data)

    labels = ushapelet.predict(data)
    assert labels.shape[0] == len(data), "Labels should match data length."

    unique_labels = np.unique(labels)
    assert (
        len(unique_labels) <= 2
    ), "Expected up to 2 distinct labels in single-shapelet mode."

    assert (
        ushapelet.best_shapelet_ is not None
    ), "Expected a best shapelet for 2-cluster scenario."
    assert ushapelet.best_gap_ != -float("inf"), "Best gap should be finite."
    assert ushapelet.best_split_ is not None, "Best split should be non-null."
    assert (
        isinstance(ushapelet.best_loc_, tuple) and len(ushapelet.best_loc_) == 2
    ), "Location must be a (series_index, start_index) tuple."


def test_ushapelet_clusterer_three_real_clusters():
    """Test UShapeletClusterer with n_clusters=3 (multi-shapelet + k-means)."""
    data = create_synthetic_data_three_clusters()

    # multi-shapelet approach with 3 clusters
    ushapelet = UShapeletClusterer(
        subseq_length=3,
        sax_length=4,
        projections=5,
        lb=1,
        ub=8,  # large enough to allow many splits
        n_clusters=3,
        random_state=42,
    )
    ushapelet.fit(data)
    labels = ushapelet.predict(data)

    assert labels.shape[0] == len(data), "Labels should match data length."
    unique_labels = np.unique(labels)
    # We have 3 distinct levels, so we expect atleast 2 or 3 clusters. Ideally 3.
    assert 2 <= len(unique_labels) <= 3, (
        "Expected 2 or 3 distinct labels in 3-cluster scenario. Found "
        f"{len(unique_labels)}."
    )
