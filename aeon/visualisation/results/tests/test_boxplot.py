"""Tests for the boxplot plotting function."""

import os

import pytest

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_boxplot

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "testing/example_results_files/",
)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_boxplot():
    """Test plot_boxplot."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]

    data = univariate_equal_length
    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )

    fig, ax = plot_boxplot(res, cls, plot_type="violin")
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    fig, ax = plot_boxplot(res, cls, plot_type="boxplot", outliers=False)
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    fig, ax = plot_boxplot(res, cls, plot_type="swarm")
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    fig, ax = plot_boxplot(res, cls, plot_type="strip")
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    with pytest.raises(ValueError):
        plot_boxplot(res, cls, plot_type="error")
