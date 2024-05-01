"""Tests for check_estimator."""

__maintainer__ = ["MatthewMiddlehurst"]

import pytest

from aeon.testing.estimator_checks import check_estimator
from aeon.testing.mock_estimators import (
    MockClassifier,
    MockClassifierMultiTestParams,
    MockForecaster,
    MockMultivariateSeriesTransformer,
)
from aeon.transformations.exponent import ExponentTransformer

EXAMPLE_CLASSES = [
    MockClassifier,
    MockForecaster,
    MockMultivariateSeriesTransformer,
    MockClassifierMultiTestParams,
]


@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_passed(estimator_class):
    """Test that check_estimator returns only passed tests for examples we know pass."""
    estimator = estimator_class.create_test_instance()

    result_class = check_estimator(estimator_class, verbose=False)
    assert all(x == "PASSED" for x in result_class.values())

    result_instance = check_estimator(estimator, verbose=False)
    assert all(x == "PASSED" for x in result_instance.values())

    # test that no exceptions are raised
    check_estimator(estimator_class, raise_exceptions=True, verbose=False)
    check_estimator(estimator, raise_exceptions=True, verbose=False)


def test_check_estimator_subset_tests():
    """Test that subsetting by tests_to_run and tests_to_exclude works as intended."""
    tests_to_run = [
        "check_get_params",
        "check_set_params",
        "check_clone",
    ]
    tests_to_exclude = ["check_set_params"]

    expected_tests = set(tests_to_run).difference(tests_to_exclude)

    results = check_estimator(
        ExponentTransformer,
        verbose=False,
        tests_to_run=tests_to_run,
        tests_to_exclude=tests_to_exclude,
    )
    results_tests = {x.split("[")[0] for x in results.keys()}

    assert results_tests == expected_tests
