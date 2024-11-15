"""Tests for the critical difference diagram maker."""

import os

import numpy as np
import pytest

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation.results._critical_difference import (
    _build_cliques,
    plot_critical_difference,
)

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "testing/example_results_files/",
)

test_clique1 = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ]
)
correct_clique1 = np.array(
    [
        [False, False, True, True],
    ]
)
test_clique2 = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ]
)
correct_clique2 = np.array([[False, False, True, True]])

test_clique3 = np.array(
    [
        [True, True, True, True, True, False, False, False, False, False, False],
        [False, True, True, True, True, True, True, False, False, False, False],
        [False, False, True, True, True, True, False, False, False, False, False],
        [False, False, False, True, True, True, True, True, True, True, True],
        [False, False, False, False, True, True, True, True, True, True, False],
        [False, False, False, False, False, True, True, True, True, True, True],
        [False, False, False, False, False, False, True, True, True, True, False],
        [False, False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, False, False, True],
    ]
)
correct_clique3 = np.array(
    [
        [True, True, True, True, True, False, False, False, False, False, False],
        [False, True, True, True, True, True, True, False, False, False, False],
        [False, False, False, True, True, True, True, True, True, True, True],
    ]
)
test_clique4 = np.array(
    [
        [True, True, True, True],
        [False, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ]
)
correct_clique4 = np.array([[True, True, True, True]])


CLIQUES = [
    (test_clique1, correct_clique1),
    (test_clique2, correct_clique2),
    (test_clique3, correct_clique3),
    (test_clique4, correct_clique4),
]


@pytest.mark.parametrize("cliques", CLIQUES)
def test__build_cliques(cliques):
    """Test build cliques with different cases."""
    obs_clique = _build_cliques(cliques[0])
    assert np.all(obs_clique == cliques[1])


def test__build_cliques_empty():
    """Test build cliques with empty return."""
    test_clique = np.array([[True, False], [False, True]])
    obs_clique = _build_cliques(test_clique)
    assert obs_clique.size == 0


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("correction", ["bonferroni", "holm", None])
def test_plot_critical_difference(correction):
    """Test plot critical difference diagram."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    fig, ax = plot_critical_difference(
        res,
        cls,
        correction=correction,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_critical_difference_p_values():
    """Test plot critical difference diagram."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    fig, ax, p_values = plot_critical_difference(
        res,
        cls,
        highlight=None,
        lower_better=False,
        alpha=0.05,
        width=6,
        textspace=1.5,
        reverse=True,
        return_p_values=True,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    assert isinstance(p_values, np.ndarray)
