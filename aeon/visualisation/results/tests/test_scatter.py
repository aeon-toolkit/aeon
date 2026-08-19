"""Tests for the scatter diagram makers."""

import os
from random import uniform

import numpy as np
import pytest

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import (
    plot_pairwise_scatter,
    plot_scatter_predictions,
    plot_score_vs_time_scatter,
)

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "testing/example_results_files/",
)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_pairwise_scatter():
    """Test plot_pairwise_scatter."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    cls = ["HC2", "FreshPRINCE"]

    data = univariate_equal_length
    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig, ax = plot_pairwise_scatter(
        res[0], res[1], cls[0], cls[1], metric="accuracy", statistic_tests=True
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    cls = ["InceptionTime", "WEASEL-D"]

    data = univariate_equal_length
    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig, ax = plot_pairwise_scatter(
        res[0], res[1], cls[0], cls[1], metric="accuracy", title="test"
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    cls = ["InceptionTime", "WEASEL-D"]

    data = univariate_equal_length
    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    fig, ax = plot_pairwise_scatter(
        1 - res[0],
        1 - res[1],
        cls[0],
        cls[1],
        metric="error",
        statistic_tests=True,
        lower_better=True,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    fig, ax = plot_pairwise_scatter(
        1 - res[0],
        1 - res[1],
        cls[0],
        cls[1],
        metric="error",
        statistic_tests=False,
        lower_better=True,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    # best_on_top = False (reversed ordering)
    fig_false, ax_false = plot_pairwise_scatter(
        res[0],
        res[1],
        cls[0],
        cls[1],
        metric="accuracy",
        title="Test Plot best_on_top False",
        best_on_top=False,
    )
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig_false, plt.Figure) and isinstance(ax_false, plt.Axes)

    # Test error handling for metrics
    with pytest.raises(ValueError):
        plot_pairwise_scatter(
            res[0], res[1], cls[0], cls[1], metric="error", lower_better=False
        )
    with pytest.raises(ValueError):
        plot_pairwise_scatter(
            res[0], res[1], cls[0], cls[1], metric="accuracy", lower_better=True
        )


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_scatter_predictions():
    """Test whether plot_scatter_predictions runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    y = np.array([uniform(0.0, 10.0) for _ in range(100)])
    y_pred = y + np.array([uniform(-1, 1) for _ in range(100)])

    method = "new method"
    dataset = "toy"

    fig, ax = plot_scatter_predictions(y, y_pred, title=f"{method}-{dataset}")
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_score_vs_time_scatter():
    """Test whether plot_score_vs_time_scatter runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    fig, ax = plot_score_vs_time_scatter(
        [0.9, 0.7, 0.5, 0.3],
        [9000, 4000, 1500, 100],
        names=["Classifier 1", "Classifier 2", "Classifier 3", "Classifier 4"],
        title="Score vs Time",
        log_time=True,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
