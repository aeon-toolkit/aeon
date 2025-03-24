"""Tests for check_estimator."""

__maintainer__ = ["MatthewMiddlehurst"]

import pytest

from aeon.clustering import TimeSeriesKMeans
from aeon.testing.estimator_checking import check_estimator, parametrize_with_checks
from aeon.testing.estimator_checking._estimator_checking import _get_check_estimator_ids
from aeon.testing.mock_estimators import (
    MockClassifier,
    MockClassifierParams,
    MockRegressor,
    MockSegmenter,
)
from aeon.testing.mock_estimators._mock_anomaly_detectors import MockAnomalyDetector
from aeon.testing.utils.deep_equals import deep_equals
from aeon.transformations.collection import Normalizer

test_classes = [
    MockClassifier,
    MockRegressor,
    TimeSeriesKMeans,
    MockSegmenter,
    MockAnomalyDetector,
    # MockMultivariateSeriesTransformer,
    Normalizer,
    MockClassifierParams,
]
test_classes = {c.__name__: c for c in test_classes}


@parametrize_with_checks(list(test_classes.values()), use_first_parameter_set=True)
def test_parametrize_with_checks_classes(check):
    """Test parametrize_with_checks with class input."""
    name = (
        _get_check_estimator_ids(check)
        .split("=")[1]
        .split(",")[0]
        .split("(")[0]
        .split(")")[0]
    )
    assert callable(check)
    dict_before = test_classes[name].__dict__.copy()
    dict_before.pop("__slotnames__", None)
    check()
    dict_after = test_classes[name].__dict__.copy()
    dict_after.pop("__slotnames__", None)
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg


test_instances = [c._create_test_instance() for c in list(test_classes.values())]
test_instances = {c.__class__.__name__: c for c in test_instances}


@parametrize_with_checks(list(test_instances.values()), use_first_parameter_set=True)
def test_parametrize_with_checks_instances(check):
    """Test parametrize_with_checks with estimator instance input."""
    name = (
        _get_check_estimator_ids(check)
        .split("=")[1]
        .split(",")[0]
        .split("(")[0]
        .split(")")[0]
    )
    assert callable(check)
    dict_before = test_instances[name].__dict__.copy()
    check()
    dict_after = test_instances[name].__dict__.copy()
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg


@pytest.mark.parametrize("estimator_class", list(test_classes.values()))
def test_check_estimator_passed(estimator_class):
    """Test that check_estimator returns only passed tests for examples we know pass."""
    estimator = estimator_class._create_test_instance()

    result_class = check_estimator(estimator_class, verbose=False)
    assert all(x == "PASSED" for x in result_class.values())

    result_instance = check_estimator(estimator, verbose=False)
    assert all(x == "PASSED" for x in result_instance.values())

    # test that no exceptions are raised
    dict_before = estimator_class.__dict__.copy()
    dict_before.pop("__slotnames__", None)
    check_estimator(estimator_class, raise_exceptions=True, verbose=False)
    dict_after = estimator_class.__dict__.copy()
    dict_after.pop("__slotnames__", None)
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg

    dict_before = estimator.__dict__.copy()
    check_estimator(estimator, raise_exceptions=True, verbose=False)
    dict_after = estimator.__dict__.copy()
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg


def test_check_estimator_subset_tests():
    """Test that subsetting by tests_to_run and tests_to_exclude works as intended."""
    tests_to_run = [
        "check_get_params",
        "check_set_params",
        "check_inheritance",
    ]
    tests_to_exclude = ["check_set_params"]

    expected_tests = [
        "check_inheritance(estimator_class=MockClassifier)",
        "check_get_params(estimator=MockClassifier())",
    ]

    results = check_estimator(
        MockClassifier,
        verbose=False,
        checks_to_run=tests_to_run,
        checks_to_exclude=tests_to_exclude,
    )
    results_tests = [x for x in results.keys()]

    assert results_tests == expected_tests
