"""Test the plotting functions for segmentation."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_pandas_series
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_series_with_change_points


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_with_change_points():
    """Test whether plot_series_with_change_points runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    series = make_example_pandas_series(n_timepoints=50)
    chp = np.random.randint(0, len(series), 3)

    fig, ax = plot_series_with_change_points(series, chp)
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    plt.close()
