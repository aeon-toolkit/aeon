# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Critical Difference maker."""
import os

import numpy as np

# import pytest
# from pytest import raises
from scipy.stats import rankdata

from aeon.benchmarking._critical_difference import (
    _check_friedman,
    build_cliques,
    nemenyi_cliques,
    plot_critical_difference,
    wilcoxon_holm_cliques,
)
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_data_lists import univariate_equal_length


def test__check_friedman():
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    alpha = 0.05
    data = univariate_equal_length
    data_path = (
        "../example_results"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../example_results/"
    )

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    ranked_data = rankdata(-1 * res, axis=1)

    is_significant = _check_friedman(
        len(cls), len(univariate_equal_length), ranked_data, alpha
    )
    assert is_significant

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

    is_significant = _check_friedman(
        len(cls), len(univariate_equal_length), ranked_data, alpha
    )
    assert not is_significant


def test_nemenyi_cliques():
    pass


def test_wilcoxon_holm_cliques():
    pass


def test_build_cliques():
    pass


def test_plot_critical_difference():
    pass


if __name__ == "__main__":
    test__check_friedman()
