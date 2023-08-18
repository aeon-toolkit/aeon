# -*- coding: utf-8 -*-
"""Tests for time series k-shapes."""
import pytest

from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
from aeon.utils._testing.collection import make_3d_test_data

#    _get_cluster_values
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_plot_cluster_algorithm():
    """Test plot clusters by simply calling it."""
    X, y = make_3d_test_data(n_cases=10)

    k_medoids = TimeSeriesKMedoids(
        n_clusters=2,  # Number of desired centers
        init_algorithm="random",  # Center initialisation technique
        max_iter=10,  # Maximum number of iterations for refinement on training set
        distance="msm",  # Distance metric to use
        random_state=1,
        method="alternate",
    )
    k_medoids.fit(X)
    plot_cluster_algorithm(k_medoids, X, k_medoids.n_clusters)
