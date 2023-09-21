# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the critical difference diagram maker."""
import os

import numpy as np

# import pytest
# from pytest import raises
from scipy.stats import rankdata

from aeon.benchmarking._critical_difference import (
    _check_friedman,
    build_cliques,
    nemenyi_cliques,
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

    is_significant = _check_friedman(len(cls), len(data), ranked_data, alpha)
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

    is_significant = _check_friedman(len(cls), len(data), ranked_data, alpha)
    assert not is_significant


def test_build_cliques():
    cliques = np.array(
        [
            [True, False, True, False],
            [False, True, True, True],
            [True, True, True, False],
            [False, True, True, True],
        ]
    )

    cliques = build_cliques(cliques)

    cliques_correct = np.array(
        [
            [False, True, True, True],
            [True, True, True, False],
        ]
    )

    assert np.all(cliques == cliques_correct)

    cliques = np.array(
        [
            [True, False, False, False],
            [False, True, True, True],
            [False, True, True, False],
            [False, False, False, True],
        ]
    )

    cliques = build_cliques(cliques)

    cliques_correct = np.array(
        [
            [False, True, True, True],
        ]
    )

    assert np.all(cliques == cliques_correct)

    cliques = np.array(
        [
            [True, True, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, True, True, True, False, False, False],
            [False, False, True, True, True, True, False, False],
            [False, False, True, True, True, True, False, False],
            [False, False, False, True, True, True, False, False],
            [False, False, False, False, False, False, True, False],
            [False, False, False, False, False, False, False, True],
        ]
    )

    cliques = build_cliques(cliques)

    cliques_correct = np.array(
        [
            [True, True, False, False, False, False, False, False],
            [False, False, True, True, True, True, False, False],
        ]
    )

    assert np.all(cliques == cliques_correct)

    cliques = np.array(
        [
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [False, True, True, True, False, False],
            [False, False, True, True, False, True],
            [False, False, False, False, True, True],
            [False, False, False, True, True, True],
        ]
    )

    cliques = build_cliques(cliques)

    cliques_correct = np.array(
        [
            [True, True, True, False, False, False],
            [False, True, True, True, False, False],
            [False, False, True, True, False, True],
        ]
    )

    assert np.all(cliques == cliques_correct)

    cliques = np.array(
        [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, True],
        ]
    )

    cliques = build_cliques(cliques)

    cliques_correct = np.ndarray([])
    assert np.all(cliques == cliques_correct)


def test_nemenyi_cliques():
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    alpha = 0.05
    data_full = list(univariate_equal_length)
    data_full.sort()
    data_path = (
        "../example_results"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../example_results/"
    )

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)

    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])

    cliques = nemenyi_cliques(len(cls), len(data_full), avranks, alpha)

    assert np.all(cliques == [False, True, True, True])

    # to check the existence of a clique we select a subset of the datasets.
    data = data_full[45:55]

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)
    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])

    cliques = nemenyi_cliques(len(cls), len(data), avranks, 0.01)

    assert np.all(cliques == [True, True, True, True])


def test_wilcoxon_holm_cliques():
    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()
    data_path = (
        "../example_results"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../example_results/"
    )
    data = data_full[:50]

    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )

    ranked_data = rankdata(-1 * res, axis=1)
    avranks = ranked_data.mean(axis=0)
    avranks = np.array([s for s, _ in sorted(zip(avranks, cls))])

    cliques = wilcoxon_holm_cliques(res, cls, avranks, 0.01)
    assert np.all(cliques == [True, True, True, True])


def test_plot_critical_difference():
    pass

