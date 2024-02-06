"""Tests for the critical difference diagram maker."""

import os

import numpy as np
import pytest

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation.results._significance import _build_cliques, plot_significance

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "benchmarking/example_results/",
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


test_clique5 = np.array(
    [
        [True, True, False, True],
        [False, True, False, True],
        [False, False, False, True],
        [False, True, False, True],
    ]
)
correct_clique5 = np.array([[True, True, False, True]])

test_clique6 = np.array(
    [
        [True, True, False, True, True, True, True],
        [False, True, True, True, True, False, False],
        [False, False, True, True, True, False, False],
        [False, False, False, True, False, True, False],
    ]
)
correct_clique6 = np.array(
    [
        [True, True, False, True, True, True, True],
        [False, True, True, True, True, False, False],
    ]
)

CLIQUES = [
    (test_clique1, correct_clique1),
    (test_clique2, correct_clique2),
    (test_clique3, correct_clique3),
    (test_clique4, correct_clique4),
    (test_clique5, correct_clique5),
    (test_clique6, correct_clique6),
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
def test_plot_significance():
    """Test plot critical difference diagram."""
    _check_soft_dependencies("matplotlib")
    from matplotlib.figure import Figure

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    plot = plot_significance(
        res,
        cls,
        lower_better=False,
        alpha=0.05,
        reverse=True,
        test="wilcoxon",
        correction="holm",
        fontsize=12,
    )
    assert isinstance(plot, Figure)
