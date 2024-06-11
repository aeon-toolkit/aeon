"""Tests for collection plotting."""

import numpy as np
import pytest

from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_collection_by_class, plot_series_collection

data_to_test = [make_example_3d_numpy(), make_example_3d_numpy_list()]


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("data", data_to_test)
def test_plot_series_collection(data):
    """Test whether plot_series_collection runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    fig, ax = plot_series_collection(data[0])
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    # Test with data labels specified
    fig, ax = plot_series_collection(data[0], y=data[1])
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("data", data_to_test)
def test_plot_collection_by_class(data):
    """Test whether plot_series_collection runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    fig, ax = plot_collection_by_class(data[0], data[1])
    plt.gcf().canvas.draw_idle()

    assert (
        isinstance(fig, plt.Figure)
        and isinstance(ax, np.ndarray)
        and all([isinstance(ax_, plt.Axes) for ax_ in ax])
    )

    plt.close()
