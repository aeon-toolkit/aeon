# -*- coding: utf-8 -*-
"""Result loading tests."""
from pytest import raises

from aeon.benchmarking.results_loaders import (
    estimator_alias,
    get_estimator_results,
    get_estimator_results_as_array,
)

cls = ["HC2"]
data = ["Chinatown"]


def test_get_estimator_results():
    """Test loading results returned in a dict.

    Tests with baked in examples to avoid reliance on external website.
    """
    res = get_estimator_results(
        estimators=cls, datasets=data, path="../example_results/"
    )
    assert res["HC2"]["Chinatown"] == 0.9825072886297376


def test_get_estimator_results_as_array():
    """Test loading results returned in an array.

    Tests with baked in examples to avoid reliance on external website.
    """
    res = get_estimator_results_as_array(
        estimators=cls, datasets=data, path="../example_results/"
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
