"""Tests for the boxplot plotting function."""

import os

import pytest

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_boxplot_median

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "benchmarking/example_results/",
)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_boxplot_median():
    from matplotlib.figure import Figure

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]

    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )

    fig = plot_boxplot_median(res, cls, plot_type="violin")
    assert isinstance(fig, Figure)
    fig = plot_boxplot_median(res, cls, plot_type="boxplot", outliers=False)
    assert isinstance(fig, Figure)
    fig = plot_boxplot_median(res, cls, plot_type="swarm")
    assert isinstance(fig, Figure)
    fig = plot_boxplot_median(res, cls, plot_type="strip")
    assert isinstance(fig, Figure)

    with pytest.raises(ValueError):
        plot_boxplot_median(res, cls, plot_type="error")
