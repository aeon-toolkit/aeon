"""Tests for UShapeletClusterer."""

import numpy as np

from aeon.clustering._u_shapelet import UShapeletClusterer

# For UShapeletClusterer we ensure that after fitting the best shapelet is valid,
# that predictions return only two cluster labels, and that get_best_u_shapelet returns
# valid outputs.
expected_label_values = [0, 1]  # Only two clusters are expected.


def create_synthetic_data():
    """
    Create a synthetic dataset of univariate time series.

    Cluster 0: Lower-valued time series.
    Cluster 1: Higher-valued time series.
    """
    # Three series in cluster 0
    cluster0 = [np.array([1, 1, 2, 2, 3, 3]) for _ in range(3)]
    # Three series in cluster 1
    cluster1 = [np.array([8, 8, 9, 9, 10, 10]) for _ in range(3)]
    return cluster0 + cluster1


def test_ushapelet_clusterer():
    """Test implementation of UShapeletClusterer."""
    data = create_synthetic_data()

    ushapelet = UShapeletClusterer(
        subseq_length=3,
        sax_length=4,
        projections=5,
        lb=1,
        ub=4,
        random_state=42,
    )

    ushapelet.fit(data)

    assert (
        ushapelet.best_shapelet_ is not None
    ), "Best shapelet should not be None after fit."
    assert ushapelet.best_gap_ != -float("inf"), "Best gap should be finite."

    labels = ushapelet.predict(data)
    assert isinstance(labels, np.ndarray), "Predictions should be a numpy array."
    assert labels.shape[0] == len(data), "There should be one label per series."
    unique_labels = np.unique(labels)
    for label in unique_labels:
        assert (
            label in expected_label_values
        ), f"Label {label} is not expected; should be 0 or 1."

    best_shapelet, best_gap, best_split, best_loc = ushapelet.get_best_u_shapelet(
        data,
        ushapelet.subseq_length,
        ushapelet.sax_length,
        ushapelet.projections,
        ushapelet.lb,
        ushapelet.ub,
    )

    assert best_shapelet is not None, "Best shapelet should not be None."
    assert best_gap != -float("inf"), "Best gap should be a finite value."
    assert best_split is not None, "Best split should not be None."
    assert (
        isinstance(best_loc, tuple) and len(best_loc) == 2
    ), "Best location must be a tuple (series index, start index)."
