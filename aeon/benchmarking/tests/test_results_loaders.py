# -*- coding: utf-8 -*-
"""Result loading tests."""
import os

import pytest
from pytest import raises

from aeon.benchmarking.results_loaders import (
    NAME_ALIASES,
    estimator_alias,
    get_estimator_results,
    get_estimator_results_as_array,
)

cls = ["HC2", "FreshPRINCE", "InceptionT"]
data = ["Chinatown", "Tools"]
data_path = (
    "./aeon/benchmarking/example_results/"
    if os.getcwd().split("\\")[-1] != "tests"
    else "../example_results/"
)


def test_get_estimator_results():
    """Test loading results returned in a dict.

    Tests with baked in examples to avoid reliance on external website.
    """
    res = get_estimator_results(estimators=cls, datasets=data, path=data_path)
    assert res["HC2"]["Chinatown"] == 0.9825072886297376


def test_get_estimator_results_as_array():
    """Test loading results returned in an array.

    Tests with baked in examples to avoid reliance on external website.
    """
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path=data_path, include_missing=True
    )
    assert res[0][0] == 0.9825072886297376


def test_alias():
    """Test the name aliasing."""
    name = estimator_alias("HIVECOTEV2")
    name2 = estimator_alias("HC2")
    assert name == "HC2" and name2 == "HC2"
    name = estimator_alias("FP")
    name2 = estimator_alias("FreshPRINCEClassifier")
    assert name == "FreshPRINCE" and name2 == "FreshPRINCE"
    name = estimator_alias("WEASEL-D")
    name2 = estimator_alias("WEASEL")
    assert name == "WEASEL-Dilation" and name2 == "WEASEL-Dilation"
    with raises(ValueError):
        estimator_alias("NotAClassifier")


"""Tests for the results loaders that should not be part of the general CI."""


@pytest.mark.skip(
    reason="Only run locally, this depends on " "timeseriesclassification.com"
)
def test_load_all_classifier_results():
    """Run through all classifiers in NAME_ALIASES."""
    for type in ["accuracy", "auroc", "balancedaccuracy", "nll"]:
        for name_key in NAME_ALIASES.keys():
            res, names = get_estimator_results_as_array(
                estimators=[name_key],
                include_missing=False,
                type=type,
                default_only=False,
            )
            assert res.shape[0] >= 112
            assert res.shape[1] == 1
            res = get_estimator_results_as_array(
                estimators=[name_key],
                include_missing=True,
                type=type,
                default_only=False,
            )
            from aeon.datasets.tsc_data_lists import univariate as UCR

            assert res.shape[0] == len(UCR)
            assert res.shape[1] == 1
