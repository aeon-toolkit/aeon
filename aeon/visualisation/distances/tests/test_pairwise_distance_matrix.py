"""Test pairwise distance matrix plotting."""

import numpy as np
import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_pairwise_distance_matrix


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_pairwise_distance_matrix():
    """Test whether plot_pairwise_distance_matrix runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    distance_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    a = np.array([1.0, 2.0])
    b = np.array([1.5, 2.5])
    path = [(0, 0), (1, 1)]

    ax = plot_pairwise_distance_matrix(distance_matrix, a, b, path)
    fig = plt.gcf()
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(fig.axes) > 0

    plt.close()
