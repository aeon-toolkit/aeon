"""Tests for the critical difference diagram maker."""
import os

import numpy as np
import pytest

from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.benchmarking.results_plotting import (
    plot_boxplot_median,
    plot_scatter,
    plot_scatter_predictions,
)
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies

test_path = MODULE = os.path.dirname(__file__)
data_path = os.path.join(test_path, "../example_results/")


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


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_scatter_predictions():
    from random import uniform

    from matplotlib.figure import Figure

    y = np.array([uniform(0.0, 10.0) for _ in range(100)])
    y_pred = y + np.array([uniform(-1, 1) for _ in range(100)])

    method = "new method"
    dataset = "toy"

    fig = plot_scatter_predictions(y, y_pred, title=f"{method}-{dataset}")
    assert isinstance(fig, Figure)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_scatter():
    from matplotlib.figure import Figure

    cls = ["HC2", "FreshPRINCE"]

    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig = plot_scatter(res, cls[0], cls[1], metric="accuracy", statistic_tests=True)
    assert isinstance(fig, Figure)

    cls = ["InceptionTime", "WEASEL-D"]

    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig = plot_scatter(res, cls[0], cls[1], metric="accuracy", title="test")
    assert isinstance(fig, Figure)

    cls = ["InceptionTime", "WEASEL-D"]

    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig = plot_scatter(
        1 - res,
        cls[0],
        cls[1],
        metric="error",
        statistic_tests=True,
        lower_better=True,
    )
    assert isinstance(fig, Figure)

    fig = plot_scatter(
        1 - res,
        cls[0],
        cls[1],
        metric="error",
        statistic_tests=False,
        lower_better=True,
    )
    assert isinstance(fig, Figure)

    with pytest.raises(ValueError):
        plot_scatter(res, cls[0], cls[1], metric="error", lower_better=False)
    with pytest.raises(ValueError):
        plot_scatter(res, cls[0], cls[1], metric="accuracy", lower_better=True)


def test_plot_multi_comparison_matrix():
    # to be completed when MCM is implemented
    pass
