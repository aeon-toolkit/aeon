"""Test temporal importance curve plotting."""

import numpy as np
import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_temporal_importance_curves


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_temporal_importance_curves():
    """Test whether plot_temporal_importance_curves runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    names = ["Mean", "Median"]
    curves = [np.random.rand(50), np.random.rand(50)]

    fig, ax = plot_temporal_importance_curves(curves, names)
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    plt.close()
