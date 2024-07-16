"""Test cluster plotting."""

import numpy as np
import pytest

from aeon.clustering import TimeSeriesKMeans
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_cluster_algorithm


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_cluster_algorithm():
    """Test whether plot_cluster_algorithm runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    data = make_example_3d_numpy()
    kmeans = TimeSeriesKMeans(n_clusters=2, distance="euclidean", max_iter=5)
    kmeans.fit(data[0])

    fig, ax = plot_cluster_algorithm(kmeans, data[0], 2)
    plt.gcf().canvas.draw_idle()

    assert (
        isinstance(fig, plt.Figure)
        and isinstance(ax, np.ndarray)
        and all([isinstance(ax_, plt.Axes) for ax_ in ax])
    )

    plt.close()
