"""Tests for the critical difference diagram maker."""

import os

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import rankdata

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation.results._critical_difference import (
    _build_cliques,
    _check_friedman,
    _nemenyi_test,
    _wilcoxon_test,
    plot_critical_difference,
)

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "benchmarking/example_results/",
)


def test_nemenyi_test():
    """Test Nemenyi test for differences in ranks."""
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    alpha = 0.05
    data_full = list(univariate_equal_length)
    data_full.sort()

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)

    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])

    cliques = _nemenyi_test(avranks, len(data_full), alpha)

    assert np.all(cliques == [False, True, True, True])

    # to check the existence of a clique we select a subset of the datasets.
    data = data_full[45:55]

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)
    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])
    cliques = _nemenyi_test(avranks, len(data), 0.01)

    assert np.all(cliques == [True, True, True, True])


def test_wilcoxon_test():
    """Test Wilcoxon pairwise test for multiple estimators."""
    cls = ["HC2", "InceptionT", "WEASEL-D", "FreshPRINCE"]
    data_full = list(univariate_equal_length)
    data_full.sort()
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )
    p_vals = _wilcoxon_test(res, cls)
    assert_almost_equal(p_vals[0], np.array([1.0, 0.0, 0.0, 0.0]), decimal=2)

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )
    p_vals = _wilcoxon_test(res, cls, lower_better=True)
    assert_almost_equal(p_vals[0], np.array([1, 0.99, 0.99, 0.99]), decimal=2)


def test__check_friedman():
    """Test Friedman test for overall difference in estimators."""
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data = univariate_equal_length
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    ranked_data = rankdata(-1 * res, axis=1)
    assert _check_friedman(ranked_data) < 0.05

    # test that approaches are not significantly different.
    cls = ["HC2", "HC2", "HC2"]
    res = get_estimator_results_as_array(
        estimators=cls,
        datasets=data,
        path=data_path,
        include_missing=True,
    )
    # add some tiny noise to the results.
    np.random.seed(42)
    random_values = np.random.uniform(-0.01, 0.01, size=res.shape)
    res = np.clip(res + random_values, 0, 1)
    ranked_data = rankdata(-1 * res, axis=1)
    assert _check_friedman(ranked_data) > 0.05


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
def test_plot_critical_difference():
    """Test plot critical difference diagram."""
    _check_soft_dependencies("matplotlib")
    from matplotlib.figure import Figure

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    plot = plot_critical_difference(
        res,
        cls,
        highlight=None,
        lower_better=False,
        alpha=0.05,
        width=6,
        textspace=1.5,
        reverse=True,
    )
    assert isinstance(plot, Figure)


if __name__ == "__main__":
    test_wilcoxon_test()
    test_plot_critical_difference()
