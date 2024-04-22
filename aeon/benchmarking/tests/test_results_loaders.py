"""Result loading tests."""

import os

import pandas as pd
import pytest
from pytest import raises

from aeon.benchmarking.results_loaders import (
    estimator_alias,
    get_available_estimators,
    get_bake_off_2017_results,
    get_bake_off_2021_results,
    get_bake_off_2023_results,
    get_estimator_results,
    get_estimator_results_as_array,
)
from aeon.datasets._data_loaders import CONNECTION_ERRORS
from aeon.testing.test_config import PR_TESTING

cls = ["HC2", "FreshPRINCE", "InceptionT"]
data = ["Chinatown", "Tools"]
test_path = os.path.dirname(__file__)
data_path = os.path.join(test_path, "../example_results/")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_estimator_results():
    """Test loading results returned in a dict.

    Tests with baked in examples to avoid reliance on external website.
    """
    res = get_estimator_results(estimators=cls, datasets=data, path=data_path)
    assert res["HC2"]["Chinatown"] == 0.9825072886297376
    res = get_estimator_results(
        estimators=cls, datasets=data, path=data_path, default_only=False
    )
    assert res["HC2"]["Chinatown"][0] == 0.9825072886297376
    with pytest.raises(ValueError, match="not a valid task"):
        get_estimator_results(estimators=cls, task="skipping")
    with pytest.raises(ValueError, match="not a valid type "):
        get_estimator_results(estimators=cls, measure="madness")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_estimator_results_as_array():
    """Test loading results returned in an array.

    Tests with baked in examples to avoid reliance on external website.
    """
    res = get_estimator_results_as_array(
        estimators=cls,
        datasets=data,
        path=data_path,
        include_missing=True,
        default_only=True,
    )
    assert res[0][0] == 0.9825072886297376
    res = get_estimator_results_as_array(
        estimators=cls,
        datasets=data,
        path=data_path,
        include_missing=True,
        default_only=False,
    )
    assert res[0][0] == 0.968901846452867


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


# Tests for the results loaders that should not be part of the general CI.
# Add to this list if new results are added
CLASSIFIER_NAMES = {
    "Arsenal",
    "BOSS",
    "cBOSS",
    "CIF",
    "CNN",
    "Catch22",
    "DrCIF",
    "EE",
    "FreshPRINCE",
    "FP",
    "GRAIL",
    "HC1",
    "HC2",
    "Hydra",
    "H-InceptionTime",
    "InceptionTime",
    "LiteTime",
    "MR",
    "MiniROCKET",
    "MrSQM",
    "MR-Hydra",
    "PF",
    "QUANT",
    "RDST",
    "RISE",
    "RIST",
    "ROCKET",
    "RSF",
    "R-STSF",
    "ResNet",
    "STC",
    "STSF",
    "ShapeDTW",
    "Signatures",
    "TDE",
    "TS-CHIEF",
    "TSF",
    "TSFresh",
    "WEASEL-1.0",
    "WEASEL-2.0",
    "1NN-DTW",
}


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_all_classifier_results():
    """Run through all classifiers in CLASSIFIER_NAMES."""
    for measure in ["accuracy", "auroc", "balacc", "logloss"]:
        for name_key in CLASSIFIER_NAMES:
            res, names = get_estimator_results_as_array(
                estimators=[name_key],
                include_missing=False,
                measure=measure,
                default_only=False,
            )
            assert res.shape[0] >= 112
            assert res.shape[1] == 1
            res = get_estimator_results_as_array(
                estimators=[name_key],
                include_missing=True,
                measure=measure,
                default_only=False,
            )
            from aeon.datasets.tsc_datasets import univariate as UCR

            assert res.shape[0] == len(UCR)
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


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_bake_off_2017_results():
    """Test original bake off results."""
    default_results = get_bake_off_2017_results()
    assert default_results.shape == (85, 25)
    assert default_results[0][0] == 0.6649616368286445
    assert default_results[84][24] == 0.853
    average_results = get_bake_off_2017_results(default_only=False)
    assert average_results.shape == (85, 25)
    assert average_results[0][0] == 0.6575447570332481
    assert average_results[84][24] == 0.8578933333100001


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_bake_off_2020_results():
    """Test multivariate bake off results."""
    default_results = get_bake_off_2021_results()
    assert default_results.shape == (26, 11)
    assert default_results[0][0] == 0.99
    assert default_results[25][10] == 0.775
    average_results = get_bake_off_2021_results(default_only=False)
    assert average_results.shape == (26, 11)
    assert average_results[0][0] == 0.9755555555555556
    assert average_results[25][10] == 0.8505208333333333


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it relies on external website.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_get_bake_off_2023_results():
    """Test bake off redux results."""
    default_results = get_bake_off_2023_results()
    assert default_results.shape == (112, 34)
    assert default_results[0][0] == 0.7774936061381074
    assert default_results[111][32] == 0.9504373177842566
    average_results = get_bake_off_2023_results(default_only=False)
    assert average_results.shape == (112, 34)
    assert average_results[0][0] == 0.7692242114236999
    assert average_results[111][32] == 0.9428571428571431
