"""Tests for the stats and p-values computation."""

import os

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import rankdata

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.benchmarking.stats import check_friedman, nemenyi_test, wilcoxon_test
from aeon.datasets.tsc_datasets import univariate_equal_length

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "testing/example_results_files/",
)


def test_nemenyi_test():
    """Test Nemenyi test for differences in ranks."""
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    alpha = 0.05
    data_full = list(univariate_equal_length)
    data_full.sort()

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)

    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])

    cliques = nemenyi_test(avranks, len(data_full), alpha)
    test_clique = np.array(
        [
            [True, False, False, False],
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    assert np.all(cliques == test_clique)

    # to check the existence of a clique we select a subset of the datasets.
    data = data_full[45:55]

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)
    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])
    cliques = nemenyi_test(avranks, len(data), 0.01)
    test_clique = np.array(
        [
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )

    assert np.all(cliques == test_clique)


def test_wilcoxon_test():
    """Test Wilcoxon pairwise test for multiple estimators."""
    cls = ["HC2", "InceptionT", "WEASEL-D", "FreshPRINCE"]
    data_full = list(univariate_equal_length)
    data_full.sort()
    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )
    p_vals = wilcoxon_test(res, cls)
    assert_almost_equal(p_vals[0], np.array([1.0, 0.0, 0.0, 0.0]), decimal=2)

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )
    p_vals = wilcoxon_test(res, cls, lower_better=True)
    assert_almost_equal(p_vals[0], np.array([1, 0.99, 0.99, 0.99]), decimal=2)


def test__check_friedman():
    """Test Friedman test for overall difference in estimators."""
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data = univariate_equal_length
    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    ranked_data = rankdata(-1 * res, axis=1)
    assert check_friedman(ranked_data) < 0.05

    # test that approaches are not significantly different.
    cls = ["HC2", "HC2", "HC2"]
    res, _ = get_estimator_results_as_array(
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
    assert check_friedman(ranked_data) > 0.05
