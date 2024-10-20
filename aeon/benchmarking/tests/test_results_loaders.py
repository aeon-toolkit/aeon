"""Result loading tests."""

import os

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from aeon.benchmarking.results_loaders import (
    VALID_RESULT_MEASURES,
    estimator_alias,
    get_available_estimators,
    get_estimator_results,
    get_estimator_results_as_array,
)
from aeon.datasets._data_loaders import CONNECTION_ERRORS
from aeon.testing.testing_config import PR_TESTING
from aeon.testing.utils.deep_equals import deep_equals

cls = ["HC2", "FreshPRINCE", "InceptionT"]
data = ["Chinatown", "ItalyPowerDemand", "Tools"]
test_path = os.path.dirname(__file__)
data_path = os.path.join(test_path, "../example_results/")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
@pytest.mark.parametrize(
    "path", [data_path, "http://timeseriesclassification.com/results/ReferenceResults"]
)
def test_get_estimator_results(path):
    """Test loading results returned in a dict."""
    res = get_estimator_results(cls, datasets=data, path=path)
    assert isinstance(res, dict)
    assert len(res) == 3
    assert all(len(v) == 2 for v in res.values())
    assert res["HC2"]["Chinatown"] == 0.9825072886297376

    # test resamples
    res2 = get_estimator_results(cls, datasets=data, num_resamples=30, path=path)
    assert isinstance(res2, dict)
    assert len(res2) == 3
    assert all(len(v) == 2 for v in res2.values())
    assert isinstance(res2["HC2"]["Chinatown"], np.ndarray)
    assert len(res2["HC2"]["Chinatown"]) == 30
    assert res2["HC2"]["Chinatown"][0] == 0.9825072886297376
    assert np.average(res2["HC2"]["ItalyPowerDemand"]) == 0.9630385487528345

    res3 = get_estimator_results(cls, datasets=data, num_resamples=None, path=path)
    assert deep_equals(res3, res2)

    with pytest.raises(ValueError, match="not a valid task"):
        get_estimator_results(cls, datasets=data, task="invalid")
    with pytest.raises(ValueError, match="not a valid type"):
        get_estimator_results(cls, datasets=data, measure="invalid")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
@pytest.mark.parametrize(
    "path", [data_path, "http://timeseriesclassification.com/results/ReferenceResults"]
)
def test_get_estimator_results_as_array(path):
    """Test loading results returned in an array."""
    res, names = get_estimator_results_as_array(
        cls,
        datasets=data,
        path=path,
    )
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 3)
    assert res[0][0] == 0.9825072886297376
    assert isinstance(names, list)
    assert len(names) == 2
    assert names == ["Chinatown", "ItalyPowerDemand"]

    res2, names2 = get_estimator_results_as_array(
        cls,
        datasets=data,
        path=path,
        include_missing=True,
    )
    assert isinstance(res2, np.ndarray)
    assert res2.shape == (3, 3)
    assert res2[0][0] == 0.9825072886297376
    assert np.isnan(res2[2][2])
    assert len(names2) == 3
    assert names2 == data

    # test resamples
    res3, names3 = get_estimator_results_as_array(
        cls,
        datasets=data,
        num_resamples=10,
        path=path,
        include_missing=True,
    )
    assert isinstance(res3, np.ndarray)
    assert res3.shape == (3, 3)
    assert res3[1][1] == 0.9524781341107872
    assert np.isnan(res3[2][0])
    assert names3 == names2

    res4, names4 = get_estimator_results_as_array(
        cls, datasets=data, num_resamples=None, path=path, include_missing=True
    )
    assert isinstance(res4, np.ndarray)
    assert res4.shape == (3, 3)
    assert res4[1][0] == 0.9630385487528345

    res5, names5 = get_estimator_results_as_array("HC2", datasets=None, path=path)
    assert isinstance(res5, np.ndarray)
    assert res5.shape == (112, 1)
    assert isinstance(names5, list)
    assert len(names5) == 112


def test_alias():
    """Test the name aliasing."""
    name = estimator_alias("HIVECOTEV2")
    name2 = estimator_alias("HC2")
    assert name == "HC2" and name2 == "HC2"
    name = estimator_alias("FP")
    name2 = estimator_alias("FreshPRINCEClassifier")
    assert name == "FreshPRINCE" and name2 == "FreshPRINCE"
    name = estimator_alias("WEASEL-Dilation")
    name2 = estimator_alias("WEASEL")
    assert name == "WEASEL-2.0" and name2 == "WEASEL-1.0"
    with raises(ValueError):
        estimator_alias("NotAClassifier")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_load_all_estimator_results(task):
    """Run through estimators from get_available_estimators and load results."""
    estimators = get_available_estimators(task=task, return_dataframe=False)
    for measure in VALID_RESULT_MEASURES[task]:
        for est in estimators:
            res, names = get_estimator_results_as_array(
                est,
                task=task,
                measure=measure,
            )
            assert res.shape[0] > 25
            assert res.shape[1] == 1


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_available_estimators():
    """Test the get_available_estimators function for tsc.com results."""
    with pytest.raises(ValueError, match="not available on tsc.com"):
        get_available_estimators(task="smiling")
    classifiers = get_available_estimators(task="classification")
    assert isinstance(classifiers, pd.DataFrame)
    assert classifiers.isin(["HC2"]).any().any()
    regressors = get_available_estimators(task="regression")
    assert isinstance(regressors, pd.DataFrame)
    assert regressors.isin(["DrCIF"]).any().any()
