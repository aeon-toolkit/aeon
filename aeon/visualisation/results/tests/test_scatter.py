"""Tests for the scatter diagram makers."""

import os

import numpy as np
import pytest

import aeon
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_pairwise_scatter, plot_scatter_predictions

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "benchmarking/example_results/",
)


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
    fig = plot_pairwise_scatter(
        res, cls[0], cls[1], metric="accuracy", statistic_tests=True
    )
    assert isinstance(fig, Figure)

    cls = ["InceptionTime", "WEASEL-D"]

    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig = plot_pairwise_scatter(res, cls[0], cls[1], metric="accuracy", title="test")
    assert isinstance(fig, Figure)

    cls = ["InceptionTime", "WEASEL-D"]

    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig = plot_pairwise_scatter(
        1 - res,
        cls[0],
        cls[1],
        metric="error",
        statistic_tests=True,
        lower_better=True,
    )
    assert isinstance(fig, Figure)

    fig = plot_pairwise_scatter(
        1 - res,
        cls[0],
        cls[1],
        metric="error",
        statistic_tests=False,
        lower_better=True,
    )
    assert isinstance(fig, Figure)

    with pytest.raises(ValueError):
        plot_pairwise_scatter(res, cls[0], cls[1], metric="error", lower_better=False)
    with pytest.raises(ValueError):
        plot_pairwise_scatter(res, cls[0], cls[1], metric="accuracy", lower_better=True)
